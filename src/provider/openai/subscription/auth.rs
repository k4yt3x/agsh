//! Decoders for the JWT `id_token` returned by `auth.openai.com`.
//!
//! We extract two values:
//! - `chatgpt_account_id`, sent on every Codex request as the `ChatGPT-Account-ID` header (required
//!   for subscription auth).
//! - The `exp` claim from the `access_token`, used to schedule refresh before the token expires.
//!
//! The id_token's payload nests ChatGPT-specific claims under the namespaced key
//! `https://api.openai.com/auth`, matching Codex's own parsing in
//! `temp/codex/codex-rs/login/src/token_data.rs:71-99`.
//!
//! No signature verification: the auth server's TLS handshake provides integrity for the
//! in-transit token, and the API server validates the token on every request. Local validation
//! would only catch tampering by a process that already has full access to our memory.

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde::Deserialize;

use crate::error::{MekaError, Result};

/// Pull `chatgpt_account_id` out of an OpenAI id_token JWT. Returns `None` when the claim is absent
/// (e.g. for a free-tier account with no workspace); the caller decides whether absence is fatal.
pub(super) fn extract_account_id(id_token: &str) -> Result<Option<String>> {
    let claims: IdClaims = decode_jwt_payload(id_token)?;
    Ok(claims.auth.and_then(|auth| auth.chatgpt_account_id))
}

/// Pull the `exp` (expiration) claim out of any JWT (access_token or id_token). Value is unix epoch
/// seconds. Returns `None` if the claim is missing.
pub(super) fn extract_expiration_seconds(jwt: &str) -> Result<Option<i64>> {
    let claims: StandardClaims = decode_jwt_payload(jwt)?;
    Ok(claims.exp)
}

#[derive(Deserialize)]
struct IdClaims {
    #[serde(rename = "https://api.openai.com/auth", default)]
    auth: Option<AuthClaims>,
}

#[derive(Deserialize)]
struct AuthClaims {
    #[serde(default)]
    chatgpt_account_id: Option<String>,
}

#[derive(Deserialize)]
struct StandardClaims {
    #[serde(default)]
    exp: Option<i64>,
}

fn decode_jwt_payload<T: serde::de::DeserializeOwned>(jwt: &str) -> Result<T> {
    // JWT format: header.payload.signature, three non-empty parts.
    let mut parts = jwt.split('.');
    let payload = match (parts.next(), parts.next(), parts.next()) {
        (Some(header), Some(payload), Some(signature))
            if !header.is_empty() && !payload.is_empty() && !signature.is_empty() =>
        {
            payload
        }
        _ => {
            return Err(MekaError::Provider(
                "id_token is not a well-formed JWT".to_string(),
            ));
        }
    };

    let bytes = URL_SAFE_NO_PAD.decode(payload).map_err(|error| {
        MekaError::Provider(format!("id_token base64 decode failed: {}", error))
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|error| MekaError::Provider(format!("id_token JSON decode failed: {}", error)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthesised JWT-shape string from a payload JSON object. Header and signature are
    /// placeholders; `decode_jwt_payload` only reads the middle segment, so the others just need
    /// to be non-empty.
    fn make_jwt(payload: serde_json::Value) -> String {
        let header = URL_SAFE_NO_PAD.encode(b"{\"alg\":\"none\"}");
        let payload = URL_SAFE_NO_PAD.encode(payload.to_string().as_bytes());
        let signature = URL_SAFE_NO_PAD.encode(b"signature");
        format!("{}.{}.{}", header, payload, signature)
    }

    #[test]
    fn test_extract_account_id_namespaced_claim() {
        let jwt = make_jwt(serde_json::json!({
            "sub": "user-1",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "workspace-abc",
                "chatgpt_user_id": "user-xyz"
            }
        }));
        assert_eq!(
            extract_account_id(&jwt).unwrap().as_deref(),
            Some("workspace-abc")
        );
    }

    #[test]
    fn test_extract_account_id_missing_claim() {
        let jwt = make_jwt(serde_json::json!({
            "sub": "user-1",
            "https://api.openai.com/auth": {
                "chatgpt_user_id": "user-xyz"
            }
        }));
        assert!(extract_account_id(&jwt).unwrap().is_none());
    }

    #[test]
    fn test_extract_account_id_no_auth_namespace() {
        let jwt = make_jwt(serde_json::json!({"sub": "user-1"}));
        assert!(extract_account_id(&jwt).unwrap().is_none());
    }

    #[test]
    fn test_extract_account_id_malformed_jwt_two_parts() {
        let result = extract_account_id("only.two");
        assert!(matches!(result, Err(MekaError::Provider(_))));
    }

    #[test]
    fn test_extract_account_id_malformed_jwt_empty_parts() {
        let result = extract_account_id("..");
        assert!(matches!(result, Err(MekaError::Provider(_))));
    }

    #[test]
    fn test_extract_account_id_invalid_base64() {
        let result = extract_account_id("aaa.!!!.bbb");
        assert!(matches!(result, Err(MekaError::Provider(_))));
    }

    #[test]
    fn test_extract_account_id_invalid_json() {
        let header = URL_SAFE_NO_PAD.encode(b"{}");
        let payload = URL_SAFE_NO_PAD.encode(b"not json");
        let signature = URL_SAFE_NO_PAD.encode(b"sig");
        let jwt = format!("{}.{}.{}", header, payload, signature);
        let result = extract_account_id(&jwt);
        assert!(matches!(result, Err(MekaError::Provider(_))));
    }

    #[test]
    fn test_extract_expiration_seconds_present() {
        let jwt = make_jwt(serde_json::json!({"exp": 1_700_000_000}));
        assert_eq!(
            extract_expiration_seconds(&jwt).unwrap(),
            Some(1_700_000_000)
        );
    }

    #[test]
    fn test_extract_expiration_seconds_absent() {
        let jwt = make_jwt(serde_json::json!({"sub": "user"}));
        assert!(extract_expiration_seconds(&jwt).unwrap().is_none());
    }

    #[test]
    fn test_extract_account_id_alongside_other_claims() {
        let jwt = make_jwt(serde_json::json!({
            "sub": "user-1",
            "iat": 1_700_000_000,
            "exp": 1_700_003_600,
            "https://api.openai.com/profile": {"email": "user@example.com"},
            "https://api.openai.com/auth": {
                "chatgpt_plan_type": "pro",
                "chatgpt_account_id": "ws-deadbeef",
                "chatgpt_user_id": "u-1234",
                "chatgpt_account_is_fedramp": false
            }
        }));
        assert_eq!(
            extract_account_id(&jwt).unwrap().as_deref(),
            Some("ws-deadbeef")
        );
    }
}
