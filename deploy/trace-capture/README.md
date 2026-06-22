# Trace Capture Deployments

This directory uses **Kustomize** for multi-cluster trace capture deployments.

## Structure

```
deploy/trace-capture/
├── base/                           # Cluster-agnostic manifests
│   ├── kustomization.yaml          # Base resource list
│   ├── h200-capture.yaml           # H200 deployment (no namespace)
│   ├── conformance-queue.yaml      # Kueue queue configuration
│   └── pvcs/                       # PersistentVolumeClaims
│       ├── gemma3-model-cache-pvc.yaml
│       └── ...
│
└── overlays/                       # Per-cluster configurations
    └── inference-sim/              # Cluster-specific overlay
        └── kustomization.yaml      # sets namespace
```

## Quick Start

### Deploy to inference-sim cluster
```bash
# Using kustomize directly
kustomize build deploy/trace-capture/overlays/inference-sim | kubectl apply -f -

# Or using justfile (recommended)
just capture-up              # Deploy h200-capture
just agentic-capture-up      # Deploy agentic capture
just replay-up               # Deploy offline replay
```

### Verify deployment
```bash
# Check namespace is set
kustomize build deploy/trace-capture/overlays/inference-sim | grep "namespace: inference-sim"

# Preview without applying
kustomize build deploy/trace-capture/overlays/inference-sim > /tmp/preview.yaml
```

## Creating a New Cluster Overlay

To deploy to a different cluster:

1. **Copy the inference-sim overlay**:
   ```bash
   cp -r deploy/trace-capture/overlays/inference-sim deploy/trace-capture/overlays/my-cluster
   ```

2. **Edit `kustomization.yaml`** to set your cluster's namespace:
   ```yaml
   apiVersion: kustomize.config.k8s.io/v1beta1
   kind: Kustomization

   namespace: my-namespace

   resources:
     - ../../base
   ```

3. **Test and apply**:
   ```bash
   kustomize build deploy/trace-capture/overlays/my-cluster | kubectl apply --dry-run=client -f -
   kustomize build deploy/trace-capture/overlays/my-cluster | kubectl apply -f -
   ```

Pods use the namespace default ServiceAccount unless the namespace is configured otherwise.

## What Changed

Previously, manifests used `your-namespace` and `your-gpu-serviceaccount` placeholders requiring manual `sed` replacement. Now:

- **Base manifests** have no hardcoded namespace
- **Overlays** set namespace once via kustomize
- **Justfile recipes** use `kustomize build` instead of direct `kubectl apply`

Conformance capture Jobs are generated separately from `models.toml`; set `namespace` in `[defaults]` when changing clusters.

## Original Manifests

Original manifests are preserved in `base/` with only namespace fields removed. All container images, args, volumes, and resources remain unchanged.
