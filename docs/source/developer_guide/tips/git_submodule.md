# Git Submodule for Developers

Whenever you update the submodules, say `curobo`, you need to first commit the submodule:
```bash
cd third_party/curobo
git add {files}
git commit -m "{message}"
git push
```

Then add the submodule to the main repository and commit:
```bash
cd ..
git add third_party/curobo
git commit -m "update curobo: {message}"
git push
```
