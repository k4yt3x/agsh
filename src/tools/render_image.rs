//! `render_image` tool: turns base64 image data (provided inline or read from a scratchpad entry)
//! into a multimodal Image content block so the provider can view it.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{Tool, ToolOutput};
use crate::{
    error::{MekaError, Result},
    image::{build_image_tool_output, classify_bytes},
    permission::Permission,
    provider::ToolDefinition,
    session::SessionManager,
};

pub(super) struct RenderImageTool {
    pub session_id: Arc<RwLock<Option<Uuid>>>,
    pub session_manager: SessionManager,
}

#[async_trait]
impl Tool for RenderImageTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "render_image".to_string(),
            description: "View an image from in-memory base64 bytes or a scratchpad entry. \
                          Use this after producing image data via a command pipeline (e.g. \
                          `ffmpeg ... | base64 -w0`) to see the image directly, without \
                          needing a public URL or disk write. The bytes must decode to a \
                          supported raster image: PNG/JPEG/GIF/WebP/BMP pass through, and \
                          TIFF/ICO/HDR/EXR/TGA/PNM/QOI/DDS/Farbfeld are auto-converted to \
                          PNG. Only call this when the current model supports vision."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "from_scratchpad": {
                        "type": "string",
                        "description": "Name of a scratchpad entry containing base64-encoded image bytes. Preferred for large images."
                    },
                    "base64": {
                        "type": "string",
                        "description": "Base64-encoded image bytes passed inline. Use only for small images; prefer `from_scratchpad` for larger ones."
                    }
                }
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let from_scratchpad = input.get("from_scratchpad").and_then(|v| v.as_str());
        let inline = input.get("base64").and_then(|v| v.as_str());

        let base64_text = match (from_scratchpad, inline) {
            (Some(_), Some(_)) => {
                return Ok(ToolOutput::text(
                    "Error: provide exactly one of `from_scratchpad` or `base64`, not both."
                        .to_string(),
                    true,
                ));
            }
            (None, None) => {
                return Err(MekaError::ToolExecution {
                    tool_name: "render_image".to_string(),
                    message: "missing `from_scratchpad` or `base64` parameter".to_string(),
                });
            }
            (Some(name), None) => {
                let session_id =
                    self.session_id
                        .read()
                        .await
                        .ok_or_else(|| MekaError::ToolExecution {
                            tool_name: "render_image".to_string(),
                            message: "no active session".to_string(),
                        })?;
                self.session_manager
                    .load_tool_output(session_id, name)
                    .await?
                    .ok_or_else(|| MekaError::ToolExecution {
                        tool_name: "render_image".to_string(),
                        message: format!("scratchpad entry \"{}\" not found", name),
                    })?
            }
            (None, Some(text)) => text.to_string(),
        };

        // Tools that pipe command output into the scratchpad often include a trailing newline;
        // tolerate whitespace around the base64 payload.
        let trimmed = base64_text.trim();
        let bytes = match base64::engine::general_purpose::STANDARD.decode(trimmed) {
            Ok(bytes) => bytes,
            Err(error) => {
                return Ok(ToolOutput::text(
                    format!("Error: invalid base64 input: {}", error),
                    true,
                ));
            }
        };

        let handling = classify_bytes(&bytes);
        let marker = match from_scratchpad {
            Some(name) => format!("Image rendered from scratchpad \"{}\"", name),
            None => "Image rendered from base64 input".to_string(),
        };

        // Off the runtime; same reason as `read_file` and `fetch_url`.
        tokio::task::spawn_blocking(move || build_image_tool_output(&marker, handling, &bytes))
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "render_image".to_string(),
                message: format!("image decode task failed: {}", error),
            })
    }
}

#[cfg(test)]
mod tests {
    use std::{io::Cursor, path::Path};

    use image::{ImageFormat, RgbaImage};

    use super::*;
    use crate::{provider::ToolResultContent, tools::tests::text_content};

    async fn test_manager() -> SessionManager {
        SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("open in-memory db")
    }

    fn synthesize_image_bytes(format: ImageFormat) -> Vec<u8> {
        let img = RgbaImage::from_pixel(4, 4, image::Rgba([10, 20, 30, 255]));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), format)
            .expect("encode");
        out
    }

    fn build_tool(session_manager: SessionManager, session_id: Option<Uuid>) -> RenderImageTool {
        RenderImageTool {
            session_id: Arc::new(RwLock::new(session_id)),
            session_manager,
        }
    }

    #[tokio::test]
    async fn test_render_image_base64_input_png() {
        let png = synthesize_image_bytes(ImageFormat::Png);
        let encoded = base64::engine::general_purpose::STANDARD.encode(&png);
        let tool = build_tool(test_manager().await, None);
        let output = tool
            .execute(
                serde_json::json!({ "base64": encoded }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!output.is_error);
        assert_eq!(output.content.len(), 2);
        match &output.content[1] {
            ToolResultContent::Image { source } => {
                assert_eq!(source.media_type, "image/png");
            }
            _ => panic!("expected Image block"),
        }
    }

    #[tokio::test]
    async fn test_render_image_base64_input_converts_tiff() {
        let tiff = synthesize_image_bytes(ImageFormat::Tiff);
        let encoded = base64::engine::general_purpose::STANDARD.encode(&tiff);
        let tool = build_tool(test_manager().await, None);
        let output = tool
            .execute(
                serde_json::json!({ "base64": encoded }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!output.is_error);
        match &output.content[1] {
            ToolResultContent::Image { source } => {
                assert_eq!(source.media_type, "image/png");
            }
            _ => panic!("expected Image block"),
        }
    }

    #[tokio::test]
    async fn test_render_image_scratchpad_input() {
        let png = synthesize_image_bytes(ImageFormat::Png);
        let encoded = base64::engine::general_purpose::STANDARD.encode(&png);

        let manager = test_manager().await;
        let session_id = manager.create_session(None).await.expect("create session");
        // Trailing newline mimics how command-pipe output typically lands in the scratchpad.
        manager
            .save_tool_output(session_id, "frame", &format!("{}\n", encoded))
            .await
            .expect("save");

        let tool = build_tool(manager, Some(session_id));
        let output = tool
            .execute(
                serde_json::json!({ "from_scratchpad": "frame" }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!output.is_error);
        assert!(text_content(&output).contains("frame"));
    }

    #[tokio::test]
    async fn test_render_image_missing_scratchpad_entry() {
        let manager = test_manager().await;
        let session_id = manager.create_session(None).await.expect("create session");
        let tool = build_tool(manager, Some(session_id));

        let result = tool
            .execute(
                serde_json::json!({ "from_scratchpad": "nonexistent" }),
                CancellationToken::new(),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_render_image_missing_both_inputs_errors() {
        let tool = build_tool(test_manager().await, None);
        let result = tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_render_image_both_inputs_errors() {
        let tool = build_tool(test_manager().await, None);
        let output = tool
            .execute(
                serde_json::json!({ "from_scratchpad": "a", "base64": "aGVsbG8=" }),
                CancellationToken::new(),
            )
            .await
            .expect("returns error tool output");
        assert!(output.is_error);
        assert!(text_content(&output).contains("exactly one"));
    }

    #[tokio::test]
    async fn test_render_image_invalid_base64() {
        let tool = build_tool(test_manager().await, None);
        let output = tool
            .execute(
                serde_json::json!({ "base64": "!!!not-base64!!!" }),
                CancellationToken::new(),
            )
            .await
            .expect("returns error tool output");
        assert!(output.is_error);
        assert!(text_content(&output).contains("invalid base64"));
    }

    #[tokio::test]
    async fn test_render_image_unsupported_bytes() {
        let garbage = b"not actually an image in any format";
        let encoded = base64::engine::general_purpose::STANDARD.encode(garbage);
        let tool = build_tool(test_manager().await, None);
        let output = tool
            .execute(
                serde_json::json!({ "base64": encoded }),
                CancellationToken::new(),
            )
            .await
            .expect("returns error tool output");
        assert!(output.is_error);
        assert!(text_content(&output).contains("unsupported"));
    }
}
