# Trace Capture Deployments

This directory uses **Kustomize** for multi-cluster trace capture deployments.

## Structure

```
deploy/trace-capture/
├── base/                           # Cluster-agnostic manifests
│   ├── kustomization.yaml          # Base resource list
│   ├── h200-capture.yaml           # H200 deployment (no namespace/SA)
│   ├── conformance-queue.yaml      # Kueue queue configuration
│   └── pvcs/                       # PersistentVolumeClaims
│       ├── gemma3-model-cache-pvc.yaml
│       └── ...
│
└── overlays/                       # Per-cluster configurations
    └── inference-sim/              # Cluster-specific overlay
        └── kustomization.yaml      # namespace + serviceAccount via replacements
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

# Check serviceAccount is set on all Deployments (expect 11)
kustomize build deploy/trace-capture/overlays/inference-sim | grep "serviceAccountName:" | wc -l

# Preview without applying
kustomize build deploy/trace-capture/overlays/inference-sim > /tmp/preview.yaml
```

## Creating a New Cluster Overlay

To deploy to a different cluster:

1. **Copy the inference-sim overlay**:
   ```bash
   cp -r deploy/trace-capture/overlays/inference-sim deploy/trace-capture/overlays/my-cluster
   ```

2. **Edit `kustomization.yaml`** — two values in one file:
   ```yaml
   apiVersion: kustomize.config.k8s.io/v1beta1
   kind: Kustomization

   namespace: my-namespace  # Change this

   resources:
     - ../../base

   configMapGenerator:
     - name: cluster-config
       literals:
         - serviceAccountName=my-gpu-serviceaccount  # Change this
       options:
         disableNameSuffixHash: true

   replacements:
     - source:
         kind: ConfigMap
         name: cluster-config
         fieldPath: data.serviceAccountName
       targets:
         - select:
             kind: Deployment
           fieldPaths:
             - spec.template.spec.serviceAccountName
           options:
             create: true
   ```

3. **Test and apply**:
   ```bash
   kustomize build deploy/trace-capture/overlays/my-cluster | kubectl apply --dry-run=client -f -
   kustomize build deploy/trace-capture/overlays/my-cluster | kubectl apply -f -
   ```

The overlay also emits a `cluster-config` ConfigMap (metadata only) used as the replacement source at build time.

## What Changed

Previously, manifests used `your-namespace` and `your-gpu-serviceaccount` placeholders requiring manual `sed` replacement. Now:

- **Base manifests** have no hardcoded namespace or serviceAccount
- **Overlays** set namespace once and serviceAccount once via `configMapGenerator` + `replacements`
- **Justfile recipes** use `kustomize build` instead of direct `kubectl apply`

Conformance capture Jobs are generated separately from `models.toml` (`service_account` in `[defaults]`); update that when changing clusters too.

## Original Manifests

Original manifests are preserved in `base/` with only namespace/serviceAccount fields removed. All container images, args, volumes, and resources remain unchanged.
