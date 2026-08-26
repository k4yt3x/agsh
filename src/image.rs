//! Image format handling: detects the format of fetched/loaded bytes and, when needed, transcodes
//! uncommon formats (TIFF, ICO, HDR, EXR, TGA, PNM, QOI, DDS, Farbfeld) into PNG so providers can
//! accept them as multimodal input. Also encodes payloads to base64 for the API.

use std::io::Cursor;

use base64::Engine;
use image::ImageFormat;

use crate::{
    provider::{ImageSource, ToolResultContent},
    tools::ToolOutput,
};

/// Maximum raw image bytes before base64 encoding. Keeps the resulting base64 payload under ~5
/// MB, a safe ceiling across providers.
pub(crate) const MAX_IMAGE_RAW_BYTES: usize = 3_750_000;

/// Formats a multimodal provider (Claude, OpenAI) accepts directly in an `Image` content block.
/// Anything else must be converted to PNG.
const NATIVE_FORMATS: &[ImageFormat] = &[
    ImageFormat::Png,
    ImageFormat::Jpeg,
    ImageFormat::Gif,
    ImageFormat::WebP,
    ImageFormat::Bmp,
];

/// Classification of an input image for downstream handling.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ImageHandling {
    /// Format is already provider-native; pass bytes through unchanged.
    PassThrough(ImageFormat),
    /// Format is decodable by the `image` crate; convert to PNG.
    Convert(ImageFormat),
    /// Unknown format, or the decoder isn't compiled into this build.
    Unsupported,
}

fn classify_format(format: ImageFormat) -> ImageHandling {
    if !format.reading_enabled() {
        return ImageHandling::Unsupported;
    }
    if NATIVE_FORMATS.contains(&format) {
        ImageHandling::PassThrough(format)
    } else {
        ImageHandling::Convert(format)
    }
}

/// Classify an HTTP `Content-Type` value. Strips `; charset=...` parameters, normalizes common
/// aliases that the `image` crate doesn't recognize, then delegates to
/// `ImageFormat::from_mime_type`.
pub(crate) fn classify_content_type(content_type: &str) -> ImageHandling {
    let Some(primary) = content_type.split(';').next() else {
        return ImageHandling::Unsupported;
    };
    let primary = primary.trim().to_ascii_lowercase();

    // `image::ImageFormat::from_mime_type` only accepts canonical forms, so fold a handful of
    // widely-used aliases into their canonical equivalents.
    let canonical = match primary.as_str() {
        "image/jpg" => "image/jpeg",
        "image/x-ms-bmp" => "image/bmp",
        "image/x-tiff" => "image/tiff",
        other => other,
    };

    match ImageFormat::from_mime_type(canonical) {
        Some(format) => classify_format(format),
        None => ImageHandling::Unsupported,
    }
}

/// Classify a file extension (lowercase, no leading dot).
pub(crate) fn classify_extension(extension: &str) -> ImageHandling {
    match ImageFormat::from_extension(extension) {
        Some(format) => classify_format(format),
        None => ImageHandling::Unsupported,
    }
}

/// Classify an image by sniffing its magic bytes via `image::guess_format`.
pub(crate) fn classify_bytes(bytes: &[u8]) -> ImageHandling {
    match image::guess_format(bytes) {
        Ok(format) => classify_format(format),
        Err(_) => ImageHandling::Unsupported,
    }
}

/// The most memory one image decode may allocate.
///
/// A payload meka accepts is capped at a few megabytes, but compression ratio is not bounded: a
/// small PNG can describe a 60000x60000 canvas and decode to gigabytes. `image` allocates
/// optimistically unless told otherwise, so this is the only thing standing between a crafted image
/// in a tool result and the process. 128 MiB is a 5792x5792 RGBA image, far past any real
/// screenshot or diagram.
const MAX_DECODE_ALLOC_BYTES: u64 = 128 * 1024 * 1024;

/// Decode image bytes under [`MAX_DECODE_ALLOC_BYTES`].
fn decode_with_limits(bytes: &[u8], format: ImageFormat) -> Result<image::DynamicImage, String> {
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_DECODE_ALLOC_BYTES);
    let mut reader = image::ImageReader::with_format(Cursor::new(bytes), format);
    reader.limits(limits);
    reader
        .decode()
        .map_err(|error| format!("failed to decode {:?} image: {}", format, error))
}

/// Decode arbitrary supported image bytes and re-encode as PNG.
pub(crate) fn convert_to_png(bytes: &[u8], source: ImageFormat) -> Result<Vec<u8>, String> {
    let decoded = decode_with_limits(bytes, source)?;

    let mut out = Vec::new();
    decoded
        .write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
        .map_err(|error| format!("failed to re-encode image as PNG: {}", error))?;
    Ok(out)
}

/// Read just the image dimensions without materializing pixel data. Cheap for native formats that
/// carry W×H in the header (PNG/JPEG/GIF/WebP/BMP). Used by the Claude provider's per-request
/// downscale path.
pub(crate) fn read_image_dimensions(
    bytes: &[u8],
    format: ImageFormat,
) -> Result<(u32, u32), String> {
    let reader = image::ImageReader::with_format(Cursor::new(bytes), format);
    reader
        .into_dimensions()
        .map_err(|error| format!("failed to read {:?} image dimensions: {}", format, error))
}

/// Decode `bytes`, downscale (preserving aspect ratio) if either dimension exceeds `max_dim`, and
/// re-encode as PNG. Provider-agnostic plumbing (called by the Claude provider, where Anthropic
/// enforces a 2000 px cap on multi-image requests). Other providers shouldn't need this.
pub(crate) fn downscale_to_dim_cap(
    bytes: &[u8],
    source: ImageFormat,
    max_dim: u32,
) -> Result<Vec<u8>, String> {
    // Ask the header first. An image already inside the cap needs no work at all, and decoding it
    // only to re-encode the same pixels was the common case: every request re-decoded every
    // attached image, most of which were never oversized.
    if let Ok((width, height)) = read_image_dimensions(bytes, source)
        && width <= max_dim
        && height <= max_dim
        && source == ImageFormat::Png
    {
        return Ok(bytes.to_vec());
    }

    let decoded = decode_with_limits(bytes, source)?;
    let scaled = if decoded.width() > max_dim || decoded.height() > max_dim {
        decoded.resize(max_dim, max_dim, image::imageops::FilterType::Lanczos3)
    } else {
        decoded
    };
    let mut out = Vec::new();
    scaled
        .write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
        .map_err(|error| format!("failed to re-encode image as PNG: {}", error))?;
    Ok(out)
}

/// Run the classification pipeline end-to-end: pass-through native formats, convert others to PNG,
/// enforce the byte cap. Provider-agnostic. Does NOT enforce per-axis pixel limits (Anthropic's
/// 2000 px multi-image cap is enforced separately at the Claude provider layer in
/// `src/provider/anthropic/shared.rs`, so OpenAI providers don't pay for it). Returns `(media_type,
/// bytes)`.
///
/// `hint` is what the *source* claimed the format was (a filename extension, an HTTP
/// `Content-Type`, an MCP server's `mime_type`, a client's declared MIME). It is only consulted
/// when the bytes can't be identified, because every one of those labels is guessable-wrong and the
/// providers sniff: Anthropic rejects a JPEG labelled `image/png` with a 400, and that rejection
/// lands in a `tool_result` already committed to the session, where it fails every subsequent
/// request. Deciding the media type from the bytes is what keeps a mislabel from becoming
/// unrecoverable history.
pub(crate) fn prepare_image_payload(
    hint: ImageHandling,
    bytes: &[u8],
) -> Result<(&'static str, Vec<u8>), String> {
    let handling = match classify_bytes(bytes) {
        ImageHandling::Unsupported => hint,
        sniffed => sniffed,
    };

    match handling {
        ImageHandling::PassThrough(format) => {
            if bytes.len() > MAX_IMAGE_RAW_BYTES {
                return Err(format!(
                    "image is too large ({} bytes, max {} bytes / ~5MB base64)",
                    bytes.len(),
                    MAX_IMAGE_RAW_BYTES,
                ));
            }
            Ok((format.to_mime_type(), bytes.to_vec()))
        }
        ImageHandling::Convert(format) => {
            let png = convert_to_png(bytes, format)?;
            if png.len() > MAX_IMAGE_RAW_BYTES {
                return Err(format!(
                    "converted image is too large ({} bytes, max {} bytes / ~5MB base64)",
                    png.len(),
                    MAX_IMAGE_RAW_BYTES,
                ));
            }
            Ok((ImageFormat::Png.to_mime_type(), png))
        }
        ImageHandling::Unsupported => Err("unsupported image format".to_string()),
    }
}

/// Normalize raw image bytes into a base64 [`ImageSource`] (byte cap + format conversion via
/// [`prepare_image_payload`]). Used for *input* images (e.g. an ACP client's @-mention or pasted
/// screenshot), parallel to [`build_image_tool_output`] for tool results.
pub(crate) fn prepare_image_source(
    hint: ImageHandling,
    bytes: &[u8],
) -> Result<ImageSource, String> {
    let (media_type, payload) = prepare_image_payload(hint, bytes)?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&payload);
    Ok(ImageSource {
        source_type: "base64".to_string(),
        media_type: media_type.to_string(),
        data: encoded,
    })
}

/// Decode a base64 image attachment supplied by a *client* into an [`ImageSource`]: base64-decode
/// the payload, classify it, then enforce the size cap and convert unsupported formats to PNG.
/// Returns a human-readable message on failure, suitable for surfacing back to the caller as a
/// validation error.
///
/// `declared_mime` is only a hint; [`prepare_image_source`] decides from the bytes. That matters
/// here more than anywhere else, since a client's declared type is attacker-or-typo controlled.
///
/// Shared by the ACP `session/prompt` handler and the HTTP `POST /turn` handler so both frontends
/// enforce identical limits on client-supplied images.
pub(crate) fn decode_base64_image(data: &str, declared_mime: &str) -> Result<ImageSource, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data.as_bytes())
        .map_err(|error| format!("base64 decode failed: {}", error))?;
    prepare_image_source(classify_content_type(declared_mime), &bytes)
}

/// Number of leading base64 characters that decode to enough bytes for [`image::guess_format`] to
/// identify every format it recognises: 32 chars decode to 24, and the longest signature it matches
/// is WebP's 12-byte `RIFF....WEBP`. Used where only a base64 string is on hand and decoding a
/// multi-megabyte payload purely to sniff it would be wasteful.
const SNIFF_PREFIX_CHARS: usize = 32;

/// Classify a base64 payload by its magic bytes without decoding all of it.
///
/// Base64 decodes in independent 4-character groups, so a prefix truncated to a multiple of 4
/// decodes on its own. Returns [`ImageHandling::Unsupported`] when the prefix isn't valid base64 or
/// the bytes match no known format, which callers treat the same way as any other unusable image.
pub(crate) fn classify_base64_prefix(data: &str) -> ImageHandling {
    // Bytes rather than chars: a stray multi-byte character would make a char-count prefix unsafe
    // to slice, and any non-alphabet byte fails the decode below anyway.
    let prefix: Vec<u8> = data
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .take(SNIFF_PREFIX_CHARS)
        .collect();
    let aligned = &prefix[..prefix.len() - prefix.len() % 4];
    match base64::engine::general_purpose::STANDARD.decode(aligned) {
        Ok(bytes) => classify_bytes(&bytes),
        Err(_) => ImageHandling::Unsupported,
    }
}

/// Build a two-block `ToolOutput` (text marker + multimodal Image) from raw image bytes plus a
/// pre-computed classification. Wraps `prepare_image_payload` so error paths become a text
/// `ToolOutput` with `is_error: true`. Shared by `fetch_url`, `read_file`, and `render_image`.
pub(crate) fn build_image_tool_output(
    marker: &str,
    handling: ImageHandling,
    bytes: &[u8],
) -> ToolOutput {
    let source = match prepare_image_source(handling, bytes) {
        Ok(source) => source,
        Err(message) => {
            return ToolOutput::text(format!("Error: {}: {}", marker, message), true);
        }
    };

    ToolOutput {
        content: vec![
            ToolResultContent::Text {
                text: format!("[{}]", marker),
            },
            ToolResultContent::Image { source },
        ],
        is_error: false,
        scratchpad_hint: None,
        frontend_metadata: None,
        structured: None,
    }
}

#[cfg(test)]
mod tests {
    use image::RgbaImage;

    use super::*;

    fn synthesize_image_bytes(format: ImageFormat) -> Vec<u8> {
        let img = RgbaImage::from_pixel(4, 4, image::Rgba([128, 64, 200, 255]));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), format)
            .expect("encode");
        out
    }

    fn synthesize_image_bytes_sized(format: ImageFormat, width: u32, height: u32) -> Vec<u8> {
        let img = RgbaImage::from_pixel(width, height, image::Rgba([128, 64, 200, 255]));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), format)
            .expect("encode");
        out
    }

    /// JPEG has no alpha channel, so it needs an RGB source rather than the RGBA one above.
    fn synthesize_jpeg_bytes() -> Vec<u8> {
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([128, 64, 200]));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), ImageFormat::Jpeg)
            .expect("encode");
        out
    }

    #[test]
    fn test_classify_content_type_pass_through_png() {
        assert_eq!(
            classify_content_type("image/png"),
            ImageHandling::PassThrough(ImageFormat::Png)
        );
    }

    #[test]
    fn test_classify_content_type_jpg_alias_passes_through_as_jpeg() {
        assert_eq!(
            classify_content_type("image/jpg"),
            ImageHandling::PassThrough(ImageFormat::Jpeg)
        );
    }

    #[test]
    fn test_classify_content_type_strips_params_and_case() {
        assert_eq!(
            classify_content_type("Image/PNG; charset=utf-8"),
            ImageHandling::PassThrough(ImageFormat::Png)
        );
    }

    #[test]
    fn test_classify_content_type_bmp_alias_passes_through() {
        assert_eq!(
            classify_content_type("image/x-ms-bmp"),
            ImageHandling::PassThrough(ImageFormat::Bmp)
        );
    }

    #[test]
    fn test_classify_content_type_convertible_tiff() {
        assert_eq!(
            classify_content_type("image/tiff"),
            ImageHandling::Convert(ImageFormat::Tiff)
        );
        assert_eq!(
            classify_content_type("image/x-tiff"),
            ImageHandling::Convert(ImageFormat::Tiff)
        );
    }

    #[test]
    fn test_classify_content_type_convertible_ico() {
        assert_eq!(
            classify_content_type("image/vnd.microsoft.icon"),
            ImageHandling::Convert(ImageFormat::Ico)
        );
        assert_eq!(
            classify_content_type("image/x-icon"),
            ImageHandling::Convert(ImageFormat::Ico)
        );
    }

    #[test]
    fn test_classify_content_type_unsupported() {
        assert_eq!(
            classify_content_type("image/svg+xml"),
            ImageHandling::Unsupported
        );
        assert_eq!(
            classify_content_type("image/jxl"),
            ImageHandling::Unsupported
        );
        assert_eq!(
            classify_content_type("text/html"),
            ImageHandling::Unsupported
        );
        assert_eq!(classify_content_type(""), ImageHandling::Unsupported);
    }

    #[test]
    fn test_classify_content_type_disabled_decoder() {
        // AVIF decoder is not enabled in our Cargo features, so even though the image crate knows
        // the MIME type, we should report it as Unsupported rather than trying to decode.
        assert_eq!(
            classify_content_type("image/avif"),
            ImageHandling::Unsupported
        );
    }

    #[test]
    fn test_classify_extension_native() {
        assert_eq!(
            classify_extension("png"),
            ImageHandling::PassThrough(ImageFormat::Png)
        );
        assert_eq!(
            classify_extension("jpg"),
            ImageHandling::PassThrough(ImageFormat::Jpeg)
        );
        assert_eq!(
            classify_extension("jpeg"),
            ImageHandling::PassThrough(ImageFormat::Jpeg)
        );
        assert_eq!(
            classify_extension("bmp"),
            ImageHandling::PassThrough(ImageFormat::Bmp)
        );
    }

    #[test]
    fn test_classify_extension_convertible() {
        assert_eq!(
            classify_extension("tiff"),
            ImageHandling::Convert(ImageFormat::Tiff)
        );
        assert_eq!(
            classify_extension("tif"),
            ImageHandling::Convert(ImageFormat::Tiff)
        );
        assert_eq!(
            classify_extension("ico"),
            ImageHandling::Convert(ImageFormat::Ico)
        );
        assert_eq!(
            classify_extension("tga"),
            ImageHandling::Convert(ImageFormat::Tga)
        );
    }

    #[test]
    fn test_classify_extension_unsupported() {
        assert_eq!(classify_extension("pdf"), ImageHandling::Unsupported);
        assert_eq!(classify_extension("jxl"), ImageHandling::Unsupported);
        assert_eq!(classify_extension("svg"), ImageHandling::Unsupported);
        assert_eq!(classify_extension(""), ImageHandling::Unsupported);
    }

    #[test]
    fn test_convert_bmp_to_png_roundtrip() {
        let bmp = synthesize_image_bytes(ImageFormat::Bmp);
        let png = convert_to_png(&bmp, ImageFormat::Bmp).expect("convert");
        let decoded = image::load_from_memory_with_format(&png, ImageFormat::Png).expect("decode");
        assert_eq!(decoded.width(), 4);
        assert_eq!(decoded.height(), 4);
    }

    #[test]
    fn test_convert_tiff_to_png_roundtrip() {
        let tiff = synthesize_image_bytes(ImageFormat::Tiff);
        let png = convert_to_png(&tiff, ImageFormat::Tiff).expect("convert");
        let decoded = image::load_from_memory_with_format(&png, ImageFormat::Png).expect("decode");
        assert_eq!(decoded.width(), 4);
        assert_eq!(decoded.height(), 4);
    }

    #[test]
    fn test_convert_corrupt_bytes_returns_error() {
        let result = convert_to_png(b"not a real image", ImageFormat::Png);
        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_pass_through_within_limit() {
        let bytes = vec![0u8; 128];
        let (media_type, payload) =
            prepare_image_payload(ImageHandling::PassThrough(ImageFormat::Png), &bytes)
                .expect("ok");
        assert_eq!(media_type, "image/png");
        assert_eq!(payload, bytes);
    }

    #[test]
    fn test_prepare_pass_through_oversized_errors() {
        let bytes = vec![0u8; MAX_IMAGE_RAW_BYTES + 1];
        let error = prepare_image_payload(ImageHandling::PassThrough(ImageFormat::Png), &bytes)
            .expect_err("should error");
        assert!(error.contains("too large"));
    }

    #[test]
    fn test_prepare_convert_returns_png() {
        let tiff = synthesize_image_bytes(ImageFormat::Tiff);
        let (media_type, payload) =
            prepare_image_payload(ImageHandling::Convert(ImageFormat::Tiff), &tiff).expect("ok");
        assert_eq!(media_type, "image/png");
        image::load_from_memory_with_format(&payload, ImageFormat::Png).expect("png");
    }

    #[test]
    fn test_prepare_unsupported_errors() {
        let error =
            prepare_image_payload(ImageHandling::Unsupported, b"anything").expect_err("should err");
        assert!(error.contains("unsupported"));
    }

    /// The regression this file exists to prevent: a JPEG behind a `.png` name (or a `Content-Type:
    /// image/png`) must be labelled `image/jpeg`, because Anthropic sniffs and answers 400.
    #[test]
    fn test_prepare_media_type_comes_from_bytes_not_hint() {
        let jpeg = synthesize_jpeg_bytes();
        let (media_type, payload) =
            prepare_image_payload(ImageHandling::PassThrough(ImageFormat::Png), &jpeg).expect("ok");
        assert_eq!(media_type, "image/jpeg");
        assert_eq!(payload, jpeg);
    }

    /// A hint claiming a native format doesn't skip the transcode when the bytes need one.
    #[test]
    fn test_prepare_converts_when_bytes_need_it_despite_native_hint() {
        let tiff = synthesize_image_bytes(ImageFormat::Tiff);
        let (media_type, payload) =
            prepare_image_payload(ImageHandling::PassThrough(ImageFormat::Png), &tiff).expect("ok");
        assert_eq!(media_type, "image/png");
        image::load_from_memory_with_format(&payload, ImageFormat::Png).expect("png");
    }

    #[test]
    fn test_classify_base64_prefix_identifies_format() {
        let png = base64::engine::general_purpose::STANDARD.encode(synthesize_image_bytes_sized(
            ImageFormat::Png,
            200,
            200,
        ));
        assert!(
            png.len() > SNIFF_PREFIX_CHARS,
            "fixture must be longer than the sniffed prefix"
        );
        assert_eq!(
            classify_base64_prefix(&png),
            ImageHandling::PassThrough(ImageFormat::Png)
        );
    }

    #[test]
    fn test_classify_base64_prefix_tolerates_whitespace() {
        let raw = base64::engine::general_purpose::STANDARD.encode(synthesize_jpeg_bytes());
        let wrapped = raw
            .as_bytes()
            .chunks(8)
            .map(|chunk| String::from_utf8_lossy(chunk).into_owned())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            classify_base64_prefix(&wrapped),
            ImageHandling::PassThrough(ImageFormat::Jpeg)
        );
    }

    /// Pins [`SNIFF_PREFIX_CHARS`] against whatever `image::guess_format` actually reads, rather
    /// than against my reading of its signature table. Sniffing the truncated prefix has to give
    /// the same answer as sniffing the whole payload for every format we forward; WebP is the one
    /// that matters, since its `RIFF....WEBP` check reaches furthest into the file.
    #[test]
    fn test_classify_base64_prefix_matches_full_sniff_for_every_native_format() {
        // Hand-built headers: `image` can't encode all of these, and only the magic bytes are
        // under test.
        let mut webp = b"RIFF".to_vec();
        webp.extend_from_slice(&[0u8; 4]);
        webp.extend_from_slice(b"WEBPVP8 ");
        webp.extend(std::iter::repeat_n(0u8, 64));
        let mut gif = b"GIF89a".to_vec();
        gif.extend(std::iter::repeat_n(0u8, 64));

        let payloads: Vec<(&str, Vec<u8>)> = vec![
            (
                "png",
                synthesize_image_bytes_sized(ImageFormat::Png, 64, 64),
            ),
            ("jpeg", synthesize_jpeg_bytes()),
            ("bmp", synthesize_image_bytes(ImageFormat::Bmp)),
            ("gif", gif),
            ("webp", webp),
        ];

        for (name, bytes) in payloads {
            let full = classify_bytes(&bytes);
            assert_ne!(
                full,
                ImageHandling::Unsupported,
                "{name} fixture must be recognisable at all"
            );
            let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
            assert!(
                encoded.len() > SNIFF_PREFIX_CHARS,
                "{name} fixture must exceed the sniffed prefix for this to prove anything"
            );
            assert_eq!(
                classify_base64_prefix(&encoded),
                full,
                "{name}: sniffing the first {SNIFF_PREFIX_CHARS} base64 chars must match sniffing \
                 the whole payload"
            );
        }
    }

    #[test]
    fn test_classify_base64_prefix_rejects_non_image_and_non_base64() {
        assert_eq!(
            classify_base64_prefix("BASE64DATA"),
            ImageHandling::Unsupported
        );
        assert_eq!(classify_base64_prefix("%%%%"), ImageHandling::Unsupported);
        assert_eq!(classify_base64_prefix(""), ImageHandling::Unsupported);
    }

    #[test]
    fn test_decode_base64_image_prefers_bytes_over_declared_mime() {
        let data = base64::engine::general_purpose::STANDARD.encode(synthesize_jpeg_bytes());
        let source = decode_base64_image(&data, "image/png").expect("ok");
        assert_eq!(source.media_type, "image/jpeg");
    }

    #[test]
    fn test_read_image_dimensions_png() {
        let png = synthesize_image_bytes_sized(ImageFormat::Png, 1234, 567);
        let (width, height) = read_image_dimensions(&png, ImageFormat::Png).expect("ok");
        assert_eq!((width, height), (1234, 567));
    }

    #[test]
    fn test_downscale_to_dim_cap_resizes_oversized() {
        let png = synthesize_image_bytes_sized(ImageFormat::Png, 2400, 1200);
        let out = downscale_to_dim_cap(&png, ImageFormat::Png, 2000).expect("ok");
        let decoded = image::load_from_memory_with_format(&out, ImageFormat::Png).expect("decode");
        assert!(decoded.width() <= 2000 && decoded.height() <= 2000);
        // Aspect ratio preserved (2:1).
        assert_eq!(decoded.width() / decoded.height(), 2);
    }

    #[test]
    fn test_downscale_to_dim_cap_passes_through_dimensions_when_within_cap() {
        // Always re-encodes as PNG, but dimensions match the input when already within cap.
        let png = synthesize_image_bytes_sized(ImageFormat::Png, 800, 400);
        let out = downscale_to_dim_cap(&png, ImageFormat::Png, 2000).expect("ok");
        let decoded = image::load_from_memory_with_format(&out, ImageFormat::Png).expect("decode");
        assert_eq!((decoded.width(), decoded.height()), (800, 400));
    }

    #[test]
    fn test_downscale_to_dim_cap_handles_non_native_format() {
        let bmp = synthesize_image_bytes_sized(ImageFormat::Bmp, 2400, 600);
        let png = downscale_to_dim_cap(&bmp, ImageFormat::Bmp, 2000).expect("ok");
        let decoded = image::load_from_memory_with_format(&png, ImageFormat::Png).expect("decode");
        assert!(decoded.width() <= 2000 && decoded.height() <= 2000);
    }

    #[test]
    fn test_classify_bytes_png() {
        let png = synthesize_image_bytes(ImageFormat::Png);
        assert_eq!(
            classify_bytes(&png),
            ImageHandling::PassThrough(ImageFormat::Png)
        );
    }

    #[test]
    fn test_classify_bytes_tiff() {
        let tiff = synthesize_image_bytes(ImageFormat::Tiff);
        assert_eq!(
            classify_bytes(&tiff),
            ImageHandling::Convert(ImageFormat::Tiff)
        );
    }

    #[test]
    fn test_classify_bytes_garbage_is_unsupported() {
        assert_eq!(classify_bytes(b"not an image"), ImageHandling::Unsupported);
        assert_eq!(classify_bytes(&[]), ImageHandling::Unsupported);
    }

    #[test]
    fn test_build_image_tool_output_pass_through_png() {
        let png = synthesize_image_bytes(ImageFormat::Png);
        let output = build_image_tool_output(
            "Image fetched from https://example.com/a.png",
            ImageHandling::PassThrough(ImageFormat::Png),
            &png,
        );
        assert!(!output.is_error);
        assert_eq!(output.content.len(), 2);
        match &output.content[0] {
            ToolResultContent::Text { text } => {
                assert!(text.contains("https://example.com/a.png"));
            }
            _ => panic!("first block should be Text"),
        }
        match &output.content[1] {
            ToolResultContent::Image { source } => {
                assert_eq!(source.source_type, "base64");
                assert_eq!(source.media_type, "image/png");
                let decoded = base64::engine::general_purpose::STANDARD
                    .decode(&source.data)
                    .expect("valid base64");
                assert_eq!(decoded, png);
            }
            _ => panic!("second block should be Image"),
        }
    }

    #[test]
    fn test_build_image_tool_output_oversized_returns_error() {
        let bytes = vec![0u8; MAX_IMAGE_RAW_BYTES + 1];
        let output = build_image_tool_output(
            "Image fetched from https://example.com/big.png",
            ImageHandling::PassThrough(ImageFormat::Png),
            &bytes,
        );
        assert!(output.is_error);
        let text = match &output.content[0] {
            ToolResultContent::Text { text } => text.clone(),
            _ => panic!("expected Text block"),
        };
        assert!(text.contains("too large"));
        assert!(text.contains("big.png"));
    }

    #[test]
    fn test_build_image_tool_output_converts_tiff_to_png() {
        let tiff = synthesize_image_bytes(ImageFormat::Tiff);
        let output = build_image_tool_output(
            "rendered image",
            ImageHandling::Convert(ImageFormat::Tiff),
            &tiff,
        );
        assert!(!output.is_error);
        match &output.content[1] {
            ToolResultContent::Image { source } => {
                assert_eq!(source.media_type, "image/png");
            }
            _ => panic!("expected Image block"),
        }
    }
}
