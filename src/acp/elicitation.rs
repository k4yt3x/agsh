//! Conversion between MCP's elicitation payloads and ACP's typed elicitation surface.
//!
//! MCP hands us a server-authored JSON Schema as a raw [`serde_json::Value`]; ACP wants a typed
//! [`ElicitationSchema`] built from a closed set of property kinds. The two line up well because
//! MCP restricts elicitation schemas to flat objects of primitives, which is very nearly ACP's
//! property set -- but the mapping is partial by construction, so [`to_acp_request`] returns
//! `None` for anything it cannot express rather than sending a form the client can't render.
//!
//! Both functions here are pure, which is the point of the module: the risk in this feature is the
//! schema translation, and it is testable without a live connection.

use agent_client_protocol::schema::v1::{
    BooleanPropertySchema, CreateElicitationRequest, ElicitationAcceptAction, ElicitationAction,
    ElicitationContentValue, ElicitationFormMode, ElicitationId, ElicitationMode,
    ElicitationPropertySchema, ElicitationSchema, ElicitationSessionScope, ElicitationUrlMode,
    IntegerPropertySchema, MultiSelectPropertySchema, NumberPropertySchema, SessionId,
    StringPropertySchema,
};

use crate::mcp::elicitation::{ElicitationKind, ElicitationPrompt, ElicitationResponse};

/// Build the ACP request for an MCP elicitation, or `None` when the server's schema uses something
/// ACP has no property kind for (a nested object, a tuple array, an untyped field).
///
/// Failing closed matters here: a form the client cannot render is a prompt that never resolves,
/// and the MCP call is blocked on the answer. The caller declines instead, which the server can act
/// on.
pub(super) fn to_acp_request(
    prompt: &ElicitationPrompt,
    session_id: &SessionId,
) -> Option<CreateElicitationRequest> {
    let scope = ElicitationSessionScope::new(session_id.clone());
    let mode: ElicitationMode = match &prompt.kind {
        ElicitationKind::Url { url } => {
            // The id correlates the request with a later resolution; ACP requires one and MCP has
            // no equivalent to carry through, so it is generated per request.
            ElicitationUrlMode::new(
                scope,
                ElicitationId::new(uuid::Uuid::new_v4().to_string()),
                url,
            )
            .into()
        }
        ElicitationKind::Form { schema } => {
            ElicitationFormMode::new(scope, to_acp_schema(schema)?).into()
        }
    };
    Some(CreateElicitationRequest::new(mode, prompt.message.clone()))
}

/// Translate the user's answer back into the shape the MCP handler returns to the server.
pub(super) fn from_acp_action(action: ElicitationAction) -> ElicitationResponse {
    match action {
        ElicitationAction::Accept(accept) => ElicitationResponse::Accept {
            content: accept_content(accept),
        },
        ElicitationAction::Decline => ElicitationResponse::Decline,
        ElicitationAction::Cancel => ElicitationResponse::Cancel,
        // `ElicitationAction` is explicitly future-compatible, so an unknown action is one this
        // build has never heard of. Declining is the only answer that can't be wrong.
        _ => ElicitationResponse::Decline,
    }
}

fn accept_content(accept: ElicitationAcceptAction) -> Option<serde_json::Value> {
    let content = accept.content?;
    let fields: serde_json::Map<String, serde_json::Value> = content
        .into_iter()
        .map(|(name, value)| (name, content_value_to_json(value)))
        .collect();
    Some(serde_json::Value::Object(fields))
}

fn content_value_to_json(value: ElicitationContentValue) -> serde_json::Value {
    match value {
        ElicitationContentValue::String(text) => serde_json::Value::String(text),
        ElicitationContentValue::Integer(number) => serde_json::Value::from(number),
        ElicitationContentValue::Number(number) => serde_json::Value::from(number),
        ElicitationContentValue::Boolean(flag) => serde_json::Value::Bool(flag),
        ElicitationContentValue::StringArray(items) => {
            serde_json::Value::Array(items.into_iter().map(serde_json::Value::String).collect())
        }
        // Same reasoning as `from_acp_action`: an unknown value kind can't be faithfully
        // represented, and guessing would hand the server a wrong answer. Null at least lets the
        // server's own validation reject the field rather than acting on a fabricated value; the
        // warning is what makes the omission diagnosable when a client starts sending one.
        other => {
            tracing::warn!(
                "elicitation reply contained a value kind this build cannot represent, sending \
                 null for that field: {:?}",
                other,
            );
            serde_json::Value::Null
        }
    }
}

/// Convert a server's JSON Schema into ACP's typed form schema. `None` when the schema is not a
/// flat object, or when any property uses a kind ACP has no equivalent for.
fn to_acp_schema(schema: &serde_json::Value) -> Option<ElicitationSchema> {
    if schema.get("type").and_then(|t| t.as_str()) != Some("object") {
        return None;
    }
    let properties = schema.get("properties")?.as_object()?;
    let required: Vec<&str> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|values| values.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let mut acp_schema = ElicitationSchema::new();
    if let Some(title) = schema.get("title").and_then(|t| t.as_str()) {
        acp_schema = acp_schema.title(title.to_string());
    }
    if let Some(description) = schema.get("description").and_then(|d| d.as_str()) {
        acp_schema = acp_schema.description(description.to_string());
    }
    for (name, property) in properties {
        // `property` sets the map entry and the required list together, so the two can't drift.
        acp_schema = acp_schema.property(
            name.clone(),
            to_acp_property(property)?,
            required.contains(&name.as_str()),
        );
    }
    Some(acp_schema)
}

fn to_acp_property(property: &serde_json::Value) -> Option<ElicitationPropertySchema> {
    let title = property.get("title").and_then(|t| t.as_str());
    let description = property.get("description").and_then(|d| d.as_str());

    // Only the constraints that are unambiguously 1:1 are carried. `format` is a typed ACP enum
    // whose members don't map cleanly onto arbitrary JSON Schema format strings, so it is dropped
    // rather than guessed at; the field is still collected, just without that hint.
    match property.get("type").and_then(|t| t.as_str())? {
        "string" => {
            let mut schema = StringPropertySchema::new();
            schema.title = title.map(str::to_string);
            schema.description = description.map(str::to_string);
            schema.default = property
                .get("default")
                .and_then(|d| d.as_str())
                .map(str::to_string);
            schema.min_length = property.get("minLength").and_then(json_u32);
            schema.max_length = property.get("maxLength").and_then(json_u32);
            schema.pattern = property
                .get("pattern")
                .and_then(|p| p.as_str())
                .map(str::to_string);
            schema.enum_values = property.get("enum").and_then(string_array);
            Some(schema.into())
        }
        "integer" => {
            let mut schema = IntegerPropertySchema::new();
            schema.title = title.map(str::to_string);
            schema.description = description.map(str::to_string);
            schema.default = property.get("default").and_then(|d| d.as_i64());
            schema.minimum = property.get("minimum").and_then(|m| m.as_i64());
            schema.maximum = property.get("maximum").and_then(|m| m.as_i64());
            Some(schema.into())
        }
        "number" => {
            let mut schema = NumberPropertySchema::new();
            schema.title = title.map(str::to_string);
            schema.description = description.map(str::to_string);
            schema.default = property.get("default").and_then(|d| d.as_f64());
            schema.minimum = property.get("minimum").and_then(|m| m.as_f64());
            schema.maximum = property.get("maximum").and_then(|m| m.as_f64());
            Some(schema.into())
        }
        "boolean" => {
            let mut schema = BooleanPropertySchema::new();
            schema.title = title.map(str::to_string);
            schema.description = description.map(str::to_string);
            schema.default = property.get("default").and_then(|d| d.as_bool());
            Some(schema.into())
        }
        // ACP's only array shape is a multi-select over a fixed set of strings, so an array without
        // `items.enum` has no representation.
        "array" => {
            let values = property.get("items")?.get("enum").and_then(string_array)?;
            let mut schema = MultiSelectPropertySchema::new(values);
            schema.title = title.map(str::to_string);
            schema.description = description.map(str::to_string);
            schema.min_items = property.get("minItems").and_then(|m| m.as_u64());
            schema.max_items = property.get("maxItems").and_then(|m| m.as_u64());
            schema.default = property.get("default").and_then(string_array);
            Some(schema.into())
        }
        _ => None,
    }
}

fn json_u32(value: &serde_json::Value) -> Option<u32> {
    u32::try_from(value.as_u64()?).ok()
}

fn string_array(value: &serde_json::Value) -> Option<Vec<String>> {
    let values = value.as_array()?;
    // A partially-string array would silently lose members, which for an enum means offering the
    // user fewer choices than the server allows.
    values
        .iter()
        .map(|v| v.as_str().map(str::to_string))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn session_id() -> SessionId {
        SessionId::new("sess-1")
    }

    fn form(schema: serde_json::Value) -> ElicitationPrompt {
        ElicitationPrompt {
            server_name: "srv".to_string(),
            kind: ElicitationKind::Form { schema },
            message: "fill this in".to_string(),
        }
    }

    #[test]
    fn url_elicitation_maps_to_url_mode() {
        let prompt = ElicitationPrompt {
            server_name: "calendar".to_string(),
            kind: ElicitationKind::Url {
                url: "https://example.com/oauth".to_string(),
            },
            message: "authorize meka".to_string(),
        };
        let request = to_acp_request(&prompt, &session_id()).expect("url always converts");
        assert_eq!(request.message, "authorize meka");
        match &request.mode {
            ElicitationMode::Url(url_mode) => {
                assert_eq!(url_mode.url, "https://example.com/oauth");
                assert!(
                    !url_mode.elicitation_id.0.is_empty(),
                    "ACP requires an id and MCP supplies none, so one is generated"
                );
            }
            other => panic!("expected url mode; got {:?}", other),
        }
    }

    #[test]
    fn form_carries_every_supported_property_kind() {
        let prompt = form(serde_json::json!({
            "type": "object",
            "title": "Deploy",
            "properties": {
                "name": {"type": "string", "title": "Name", "minLength": 1},
                "count": {"type": "integer", "minimum": 0, "maximum": 10},
                "ratio": {"type": "number", "default": 0.5},
                "force": {"type": "boolean", "description": "skip checks"},
                "tags": {"type": "array", "items": {"enum": ["a", "b"]}},
            },
            "required": ["name", "count"],
        }));
        let request = to_acp_request(&prompt, &session_id()).expect("all kinds are supported");
        let ElicitationMode::Form(mode) = &request.mode else {
            panic!("expected form mode");
        };
        let schema = &mode.requested_schema;

        assert_eq!(schema.title.as_deref(), Some("Deploy"));
        assert_eq!(schema.properties.len(), 5);
        assert!(matches!(
            schema.properties.get("name"),
            Some(ElicitationPropertySchema::String(_))
        ));
        assert!(matches!(
            schema.properties.get("count"),
            Some(ElicitationPropertySchema::Integer(_))
        ));
        assert!(matches!(
            schema.properties.get("ratio"),
            Some(ElicitationPropertySchema::Number(_))
        ));
        assert!(matches!(
            schema.properties.get("force"),
            Some(ElicitationPropertySchema::Boolean(_))
        ));
        assert!(matches!(
            schema.properties.get("tags"),
            Some(ElicitationPropertySchema::Array(_))
        ));

        let required = schema.required.as_ref().expect("required list");
        assert_eq!(required.len(), 2);
        assert!(required.contains(&"name".to_string()));
        assert!(required.contains(&"count".to_string()));
    }

    #[test]
    fn string_enum_becomes_single_select() {
        let prompt = form(serde_json::json!({
            "type": "object",
            "properties": {"colour": {"type": "string", "enum": ["red", "green"]}},
        }));
        let request = to_acp_request(&prompt, &session_id()).expect("converts");
        let ElicitationMode::Form(mode) = &request.mode else {
            panic!("expected form mode");
        };
        let Some(ElicitationPropertySchema::String(schema)) =
            mode.requested_schema.properties.get("colour")
        else {
            panic!("expected a string property");
        };
        assert_eq!(
            schema.enum_values.as_deref(),
            Some(["red".to_string(), "green".to_string()].as_slice())
        );
    }

    #[test]
    fn unrepresentable_schemas_are_rejected() {
        // Each of these would otherwise be sent as a form the client cannot render, blocking the
        // MCP call on a prompt that never resolves.
        let cases = [
            // Nested object: ACP has no object property kind.
            serde_json::json!({
                "type": "object",
                "properties": {"nested": {"type": "object", "properties": {}}},
            }),
            // Array of free-form strings: ACP arrays are multi-select over a fixed set.
            serde_json::json!({
                "type": "object",
                "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
            }),
            // Untyped property.
            serde_json::json!({"type": "object", "properties": {"anything": {}}}),
            // Not an object at the top level.
            serde_json::json!({"type": "string"}),
            // No properties at all.
            serde_json::json!({"type": "object"}),
        ];
        for schema in cases {
            assert!(
                to_acp_request(&form(schema.clone()), &session_id()).is_none(),
                "should have been rejected: {}",
                schema,
            );
        }
    }

    #[test]
    fn accept_maps_every_content_value_kind() {
        let mut content = BTreeMap::new();
        content.insert(
            "name".to_string(),
            ElicitationContentValue::String("ada".to_string()),
        );
        content.insert("count".to_string(), ElicitationContentValue::Integer(3));
        content.insert("ratio".to_string(), ElicitationContentValue::Number(0.25));
        content.insert("force".to_string(), ElicitationContentValue::Boolean(true));
        content.insert(
            "tags".to_string(),
            ElicitationContentValue::StringArray(vec!["a".to_string()]),
        );

        let response = from_acp_action(ElicitationAction::Accept(
            ElicitationAcceptAction::new().content(content),
        ));
        let ElicitationResponse::Accept {
            content: Some(json),
        } = response
        else {
            panic!("expected accepted content");
        };
        assert_eq!(
            json,
            serde_json::json!({
                "name": "ada",
                "count": 3,
                "ratio": 0.25,
                "force": true,
                "tags": ["a"],
            })
        );
    }

    #[test]
    fn accept_without_content_stays_empty() {
        let response = from_acp_action(ElicitationAction::Accept(ElicitationAcceptAction::new()));
        assert!(matches!(response, ElicitationResponse::Accept {
            content: None
        }));
    }

    #[test]
    fn decline_and_cancel_map_through() {
        assert!(matches!(
            from_acp_action(ElicitationAction::Decline),
            ElicitationResponse::Decline
        ));
        assert!(matches!(
            from_acp_action(ElicitationAction::Cancel),
            ElicitationResponse::Cancel
        ));
    }
}
