"""Telegram bot front-end for the nntile-gateway HTTP API.

The bot talks to the gateway over HTTP only -- it doesn't import
nntile or load any model weights itself, so it can run on a host
with no CUDA. See `infra/README.md` for deployment and the gateway
package for the server side."""
