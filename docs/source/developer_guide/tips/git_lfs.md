# Git LFS

## Add New File Patterns to `.gitattributes`

When you add new file patterns to `.gitattributes`, you may encounter the following error when running `git status`:
```text
Encountered 7 file(s) that should have been pointers, but weren't:
```

These files in previous commits are not tracked by Git LFS. You need to manually make them tracked by Git LFS.

```bash
git lfs migrate import --no-rewrite ${your_broken_file}
```

For more details, please refer to [this answer](https://stackoverflow.com/a/57820265).
