# UnicodeEncodeError - Root Cause Analysis & Solution

## Problem Summary

**Error:** `UnicodeEncodeError: 'utf-8' codec can't encode characters in position 3196-3197: surrogates not allowed`

**Location:** Starlette's HTMLResponse rendering at `/usr/local/lib/python3.10/site-packages/starlette/responses.py`, line 54

**Trigger:** GET request to `/?logs=container` endpoint

## Root Cause Analysis

### The Issue

In commit `d4759f3` (Fix Data Management tab issues), all Unicode emoji and special characters in `tabs/tab_data.py` were **intentionally double-escaped**.

**What Happened:**
- Original (Correct): `"✅ Loaded data"` - actual Unicode character
- After Commit (Incorrect): `"\\u2705 Loaded data"` - escaped backslash-u sequence

**Why This Causes the Error:**

1. The Python source file contains literal text `"\\u2705"` (backslash, u, 2, 7, 0, 5)
2. When the logger prints this, it tries to interpret `\u2705` as an escape sequence
3. Some escape sequences (like `\ud83d` and `\udcc1`) are **surrogate pairs** - halves of multi-byte emoji characters
4. When these surrogate pairs are rendered into HTML, they create invalid UTF-8 sequences
5. Starlette's HTMLResponse tries to encode the HTML as UTF-8 and fails with "surrogates not allowed"

**Examples of Problematic Sequences:**

```python
# Double-escaped (WRONG - causes error)
logger.info("\\ud83d\\udd04 User clicked Load Example Data")

# Should be (CORRECT - works properly)
logger.info("🔄 User clicked Load Example Data")
```

### Evidence from Git Diff

```diff
- logger.info("🔄 User clicked Load Example Data")
+ logger.info("\\ud83d\\udd04 User clicked Load Example Data")

- logger.info(f"✅ Successfully generated {n} records")
+ logger.info(f"\\u2705 Successfully generated {n} records")

- ui.notification_show("⚠️ Large file: showing first 100,000 rows", type="warning")
+ ui.notification_show("\\u26a0\\ufe0f Large file: showing first 100,000 rows", type="warning")
```

### Why This Was Done

It appears the double-escaping was an unintended side effect, possibly from:
1. Copy-pasting code from a formatted document or markdown
2. Using a text editor that automatically escapes Unicode
3. Running a script that incorrectly escaped the Unicode characters

## Solutions

### Solution 1: Revert to Original Unicode Characters (RECOMMENDED)

**Best for:** Production use, maintainability, readability

**Steps:**
1. Checkout the version before the problematic commit
2. Apply only the functional fixes (reactive event handlers) without the Unicode changes
3. Commit and push

```bash
# Checkout the version before d4759f3
git checkout a382bfc -- tabs/tab_data.py

# Manually apply the functional fixes:
# 1. Change @reactive.event(lambda: input.btn_load_example()) to @reactive.event(input.btn_load_example)
# 2. Add reactive.Calc functions for proper reactivity
# 3. Name all reactive functions

# Commit the fix
git add tabs/tab_data.py
git commit -m "Fix Data Management tab - proper Unicode handling"
```

### Solution 2: Use Python's encode/decode with 'surrogateescape' (TEMPORARY FIX)

**Best for:** Quick testing, understanding the issue

**Steps:**
Add error handling to the logger configuration:

```python
# In logger.py or config.py
import sys
import io

# Create a custom stdout handler that handles surrogates
class SurrogateSafeTextIOWrapper(io.TextIOWrapper):
    def write(self, text):
        try:
            return super().write(text)
        except UnicodeEncodeError:
            # Fall back to replacing surrogates
            safe_text = text.encode('utf-8', errors='replace').decode('utf-8')
            return super().write(safe_text)

# Apply to sys.stdout
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = SurrogateSafeTextIOWrapper(
        sys.stdout.buffer,
        encoding='utf-8',
        errors='surrogatepass'
    )
```

**Limitations:**
- Doesn't fix the root cause
- May cause display issues with complex emoji
- Not recommended for production

### Solution 3: Remove All Emoji from Logger Messages (WORKAROUND)

**Best for:** Maximum compatibility, minimal changes

**Steps:**
Replace all emoji with text equivalents:

```python
# Before (problematic)
logger.info("🔄 User clicked Load Example Data")

# After (safe)
logger.info("[LOAD] User clicked Load Example Data")
```

**Limitations:**
- Loss of visual clarity
- Requires extensive changes
- Doesn't address the underlying issue

## Recommended Approach

### Phase 1: Immediate Fix (Solution 1)

1. **Restore original Unicode characters:**

```python
# Create a script to fix the file
import re

with open('tabs/tab_data.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove double-escaping (this is a simplified approach)
# In practice, you should manually review and fix each line

# Examples of manual fixes needed:
# "\\u2705" -> "✅"
# "\\u274c" -> "❌"
# "\\ud83d\\udd04" -> "🔄"
# "\\u26a0\\ufe0f" -> "⚠️"
# "\\ud83d\\udcc2" -> "📂"
# "\\ud83d\\udcca" -> "📊"
# "\\ud83d\\udcbe" -> "💾"

with open('tabs/tab_data.py', 'w', encoding='utf-8') as f:
    f.write(content)
```

2. **Test the application:**

```bash
# Start the application
python -m shiny run app.py

# Access the logs endpoint
curl http://localhost:7860/?logs=container

# Verify no UnicodeEncodeError occurs
```

### Phase 2: Prevent Future Issues

1. **Add pre-commit hooks:**

```bash
# Create .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: check-unicode
        name: Check for double-escaped Unicode
        entry: python3 scripts/check_unicode.py
        language: system
        files: \.py$
```

2. **Add linting rules:**

```python
# Setup pylint or flake8 to catch these issues
# In pyproject.toml or setup.cfg:
[tool.pylint.messages_control]
disable = []
enable = [
    "W1402",  # Anomalous backslash in string
]
```

3. **Document coding standards:**

```markdown
## Unicode Character Usage Guidelines

1. Always use actual Unicode characters in source code:
   - ✅ Correct: logger.info("✅ Success")
   - ❌ Wrong: logger.info("\\u2705 Success")

2. File encoding must be UTF-8:
   - Add `# -*- coding: utf-8 -*-` at the top of files
   - Configure your editor to save as UTF-8

3. Test with emoji in log messages:
   - Ensure your logging infrastructure handles Unicode properly
   - Test with complex emoji (surrogate pairs)
```

## Testing Checklist

- [ ] Application starts without errors
- [ ] `/` endpoint loads successfully
- [ ] `/?logs=container` endpoint works without UnicodeEncodeError
- [ ] Log messages display emoji correctly
- [ ] All three Data Management tab issues remain fixed:
  - [ ] Loading spinner no longer appears continuously
  - [ ] "Load Example Data" button works
  - [ ] File upload works
- [ ] No surrogate pair errors in console logs
- [ ] Application handles Thai language characters correctly

## Additional Notes

### Understanding Surrogate Pairs

Emoji and some other Unicode characters are represented as **surrogate pairs** in UTF-16:

- High surrogate: `\ud800` to `\udbff`
- Low surrogate: `\udc00` to `\udfff`

When these appear individually in UTF-8 context, they're invalid and cause encoding errors.

**Example:**
```python
# This emoji is a surrogate pair
"🔄" = "\ud83d\udd04"

# If you double-escape it:
"\\ud83d\\udd04"  # Now these are literal backslash sequences

# When Python tries to interpret \ud83d and \udd04 individually:
# They create invalid UTF-8 surrogates → UnicodeEncodeError
```

### Python's Unicode Handling

Python 3 uses Unicode strings internally. When you have:

```python
text = "\\u2705"  # This is 7 characters: \, u, 2, 7, 0, 5
```

Python stores this as literal characters, not as the emoji. When you try to:

```python
print(text)  # Prints: \u2705 (not ✅)
```

But if this text gets into HTML that Starlette tries to encode:

```python
html = f"<div>{text}</div>"
html.encode('utf-8')  # May fail if text contains surrogates
```

This is where the error occurs.

## References

- Python Unicode Documentation: https://docs.python.org/3/howto/unicode.html
- Starlette Response Encoding: https://www.starlette.io/responses/
- UTF-8 Surrogate Pairs: https://en.wikipedia.org/wiki/UTF-16#Code_points_from_U+010000_to_U+10FFFF

---

**Document Version:** 1.0  
**Created:** 2026-01-03  
**Status:** Ready for Implementation
