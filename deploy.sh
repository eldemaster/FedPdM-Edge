#!/bin/bash
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: ./deploy.sh [--ca-cert PATH --cert PATH --key PATH | --insecure]

Deploy client code over SSH. Provision certificates separately on EACH edge node;
paths below refer to files already on that node. Never share a client private key.
  --ca-cert PATH   Trusted CA (default: FL_CA_CERT)
  --cert PATH      This node's client certificate (default: FL_CLIENT_CERT)
  --key PATH       This node's private key (default: FL_CLIENT_KEY)
  --insecure       Explicit plaintext opt-in for local development only

Example (use the same remote paths with a distinct identity on each node):
  ./deploy.sh --ca-cert /etc/fedpdm/ca.pem \
    --cert /etc/fedpdm/client.pem --key /etc/fedpdm/client.key
The script prints the client launch options; set CLIENT_ID and FL_SERVER per node.
For the server, use FL_CA_CERT, FL_SERVER_CERT, FL_SERVER_KEY (or the corresponding
server.py flags). mTLS is required by default. --no-require-client-cert explicitly
allows server-only TLS. Clients can use CA-only TLS with --ca-cert alone.
Optional client protection: DP_CLIP > 0 and DP_NOISE_MULT >= 0 (std = their product).
Clipping alone provides no DP guarantee; privacy accounting is deployment-specific.
HELP
}

ca_cert=${FL_CA_CERT:-}
client_cert=${FL_CLIENT_CERT:-}
client_key=${FL_CLIENT_KEY:-}
insecure=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ca-cert|--cert|--key)
            [[ $# -ge 2 && -n "$2" ]] || { usage >&2; exit 2; }
            case "$1" in
                --ca-cert) ca_cert=$2 ;;
                --cert) client_cert=$2 ;;
                --key) client_key=$2 ;;
            esac
            shift 2 ;;
        --insecure) insecure=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if "$insecure"; then
    [[ -z "$ca_cert$client_cert$client_key" ]] || {
        echo '--insecure conflicts with TLS paths/environment variables' >&2; exit 2;
    }
    echo 'WARNING: INSECURE plaintext transport; local development only' >&2
    transport=(--insecure)
else
    [[ -n "$ca_cert" && -n "$client_cert" && -n "$client_key" ]] || {
        echo 'Deployment requires CA, client certificate and key paths; see --help' >&2; exit 2;
    }
    transport=(--ca-cert "$ca_cert" --cert "$client_cert" --key "$client_key")
fi

for target in eldemaster@192.168.1.147 eldemaster@192.168.1.115; do
    echo "Deploying to $target..."
    # Do not copy symlinks, development environments, or conventional key/cert files.
    rsync -rtvz --exclude '.git' --exclude '.agent-office' --exclude '.agents' \
        --exclude '.codex' --exclude '.aws' --exclude 'results' --exclude 'venv' \
        --exclude '.venv' --exclude '__pycache__' --exclude '*.pem' --exclude '*.key' \
        --exclude '*.crt' --exclude '*.p12' --exclude '*.pfx' ./ "$target:~/fedpdm_client/"
done
printf 'Deployment complete. Client launch options (set CLIENT_ID and FL_SERVER):\n  python3 client.py'
printf ' %q' "${transport[@]}"
printf '\n'
