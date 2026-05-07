# HuggingFace for Developers

## SSH Config for HuggingFace

Generate a SSH key and add the following to your `~/.ssh/config` file:
```
Host hf.co
    HostName hf.co
    User git
    IdentityFile {path/to/your/ssh/key}
```

Then add the public key to your HuggingFace account.

For more information, see the [official documentation](https://huggingface.co/docs/hub/en/security-git-ssh).

## HuggingFace Token

You can generate a token [here](https://huggingface.co/settings/tokens).

The token is used to login to HuggingFace:
```bash
huggingface-cli login --token {your_token}
```

Or simply:
```bash
export HF_TOKEN={your_token}
```

For more information, see the [official documentation](https://huggingface.co/docs/hub/en/security-tokens).
