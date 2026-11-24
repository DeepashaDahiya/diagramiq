# DiagramIQ public demo

This directory is a static, lightweight showcase for Cloudflare Pages. It uses illustrative precomputed sample outputs and does not call the local Flask/Ollama inference pipeline. The original full application and local deployment workflow remain unchanged.

## Cloudflare Pages settings
- Production branch: `demo-hosting`
- Root directory: `demo-site`
- Framework preset: None
- Build command: leave blank (or `exit 0`)
- Build output directory: `.`

The `index.html` file is the complete static site.
