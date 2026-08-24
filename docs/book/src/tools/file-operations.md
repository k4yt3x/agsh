# File Operations

## `read_file`

Read the contents of a file at a given path. Supports text files and images.

**Permission:** Read

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | yes | The file path to read |
| `offset` | integer | no | Line number to start reading from (0-based) |
| `limit` | integer | no | Maximum number of lines to read (default: 2000) |
| `regex` | string | no | Return matching lines (capped, exact value advertised in the tool's parameter schema) instead of a line range. Skipped for image files. |
| `scratchpad` | string | no | Save output to the scratchpad under this name |

### Behavior

- `limit` defaults to 2000 lines. Whenever the read stops short of the end of the file, whether because of the default or an explicit `limit`, a notice naming the range shown and the total line count is appended. A definitive answer drawn from a silent truncation is worse than an error.
- Use `offset`/`limit` to page through large files.
- A single read holds at most 16 MiB in memory. Asking for the whole of a file larger than that is refused, because there is no bounded way to return it; asking for a *window* of one is not, and streams past everything outside the window. So a command-output capture larger than the ceiling stays readable a page at a time, which is what [`execute_command`](shell.md) promises when it spills one to a file.
- A read that shows the whole file returns it byte for byte, so a CRLF file stays CRLF and an `old_string` copied out of it applies as written. A windowed read normalises line endings to `\n`; if a later `edit_file` misses for that reason it says so.
- Under [ACP](../usage/acp.md) the editor is asked for the whole document and the window is applied here, so both the truncation notice and the freshness fingerprint describe the document rather than the slice.
- `regex` runs the pattern against each line and returns `line:content` rows (like `grep -n`). It bypasses `offset`/`limit` and is meaningless on image content. Under [ACP](../usage/acp.md) it searches the editor's copy of the file, like any other text read, so a search and the edit that follows it see the same document.

### Image files

Recognized image extensions are returned as base64-encoded multimodal content:

- **Provider-native** (pass-through): `.png`, `.jpg`/`.jpeg`, `.gif`, `.webp`, `.bmp`
- **Convertible** (decoded and re-encoded as PNG transparently): `.tif`/`.tiff`, `.ico`, `.hdr`, `.exr`, `.tga`, `.pbm`/`.pgm`/`.ppm`/`.pnm`, `.qoi`, `.dds`, `.ff`/`.farbfeld`
- **Unsupported** (fall through to text read, which will fail on binary): `.svg`, `.jxl`, `.heic`, `.avif`

Images are rejected if the final payload exceeds 3.75 MB (~5 MB base64). Conversion can enlarge an image, so a small TIFF may produce a too-large PNG.

Only read image files when the current model supports vision input; text-only models will either error or silently drop the image block.

### Examples

Read an entire file:

```text
meka [r] > show me the contents of src/main.rs
```

Read lines 10-20:

```text
meka [r] > show me lines 10 through 20 of src/main.rs
```

---

## `edit_file`

Modify a file. Supports two modes: **replace** (swap `old_string` for `new_string`) and **insert** (place content before or after `old_string` while preserving the anchor). The file must have been read with `read_file` first (unless `force` is set).

**Permission:** Workspace

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | yes | The file path to edit |
| `old_string` | string | yes | The exact string to find (acts as anchor in insert modes) |
| `new_string` | string | one of three | Replace mode: replacement for `old_string` (an empty string deletes it) |
| `insert_before` | string | one of three | Insert mode: text inserted immediately before `old_string` (anchor preserved) |
| `insert_after` | string | one of three | Insert mode: text inserted immediately after `old_string` (anchor preserved) |
| `replace_all` | boolean | no | Apply to every occurrence (default: false). If false and `old_string` matches more than once, the edit is rejected as ambiguous |
| `force` | boolean | no | Bypass read-before-edit requirement (default: false) |
| `scratchpad` | string | no | Save output to the scratchpad under this name |

Exactly one of `new_string`, `insert_before`, or `insert_after` must be provided. Mixing modes is rejected.

### Behavior

- If `old_string` matches more than once and `replace_all` is not set, the edit is **rejected**. Add surrounding context to make the anchor unique, or set `replace_all` to change every occurrence.
- To delete text, use replace mode with an empty `new_string`.
- The file must have been previously read with `read_file` on the same path. This prevents blind edits. Set `force` to bypass this requirement.
- The read must still be **valid**. meka records the file's modification time and size when it is read, and rejects an edit if either has changed since:

  ```text
  Error: file 'src/main.rs' changed on disk after you read it. Something else
  wrote to it (a shell command, another agent, or the user). Read it again
  before editing so you are not overwriting that change, or set force=true.
  ```

  This is a deliberately different message from the never-read case, because the next move differs: re-read to see what changed, then decide whether the edit still applies. Anything can be the other writer, an `execute_command` running `sed -i`, a [background task](../usage/background.md), or you in another window. `write_file` and a successful `edit_file` both re-record the file, so consecutive edits never trip it.

  A read served by the editor under [ACP](../usage/acp.md) is checked against the editor, not the disk. Those are two different documents that share a path: the editor serves its own copy of every file it owns, saved or not, so comparing one to the other would fire every time you save a file nobody edited and stay silent when you rewrite the buffer the agent is about to edit. meka fingerprints what the editor served and compares it against what the editor serves when the edit arrives, which it fetches anyway. Editing the buffer, or the editor reloading a file something else rewrote, is reported:

  ```text
  Error: file 'src/main.rs' changed in the editor after you read it. Someone edited
  the buffer, or the editor reloaded the file. Read it again before editing so you
  are not overwriting that change, or set force=true to edit anyway.
  ```

  Saving does not trip it: the document is unchanged, only the bytes on disk moved.
- If `old_string` is not found, the tool returns an error (without modifying the file).
- On success, the response includes a small ±3-line snippet (with line numbers, lines truncated at 200 chars) around the first edited site so you can confirm the change landed without re-reading the file.

---

## `write_file`

Create or overwrite a file with the given content.

**Permission:** Workspace

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | yes | The file path to write |
| `content` | string | yes | The content to write to the file |
| `force` | boolean | no | Overwrite a file that changed since it was read (default: false) |
| `scratchpad` | string | no | Save output to the scratchpad under this name |

### Behavior

- Creates parent directories if they do not exist.
- Overwrites the file if it already exists.
- Overwriting an **existing** file is subject to the same staleness check as `edit_file`: if the file was read and has changed since, the write is refused with the message shown above and `force` is the way past it. A whole-file rewrite is the more destructive of the two, so it is not the more permissive one. Creating a new file needs no prior read.
