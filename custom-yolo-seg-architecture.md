# Custom YOLO-seg Network Architecture

This document describes the architectural layout of the **Custom YOLO-seg** network created for single-class lane segmentation (`almondmatcha-lane-detection/segmentation`).

## Key Specifications
- **Input Resolution**: `64 x 36` RGB Image (`Batch x 3 x 36 x 64`)
- **Output Resolution**: `64 x 36` Binary Mask Logits (`Batch x 1 x 36 x 64`)
- **Target Task**: Pixel-wise lane segmentation from 1280x720 downsampled domain dataset.

---

## Mermaid Network Diagram

```mermaid
flowchart TD
    %% Node Styling
    classDef inputStyle fill:#2b5c8f,stroke:#1a385c,stroke-width:2px,color:#fff
    classDef backboneStyle fill:#1e7e34,stroke:#114b1f,stroke-width:2px,color:#fff
    classDef neckStyle fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef headStyle fill:#9333ea,stroke:#581c87,stroke-width:2px,color:#fff
    classDef outputStyle fill:#dc2626,stroke:#991b1b,stroke-width:2px,color:#fff

    %% Input
    Input["Input Image<br/>(3, 36, 64) RGB"]:::inputStyle

    %% Backbone Subgraph
    subgraph Backbone ["YOLO Backbone (Feature Extractor)"]
        Stem["Stem Conv Block<br/>(32, 36, 64)"]:::backboneStyle
        Stage1["Stage 1 Conv & ResBlock<br/>Downsample / Stride 2<br/>(64, 18, 32)"]:::backboneStyle
        Stage2["Stage 2 Conv & ResBlock<br/>Downsample / Stride 2<br/>(128, 9, 16)"]:::backboneStyle
        Stage3["Stage 3 Conv & ResBlock<br/>Downsample / Stride 2<br/>(256, 5, 8)"]:::backboneStyle
    end

    %% Neck Subgraph
    subgraph Neck ["FPN / Feature Pyramid Fusion Neck"]
        Lat4["Lateral Conv 1x1<br/>(128, 5, 8)"]:::neckStyle
        P4_Up["Upsample Bilinear<br/>to (128, 9, 16)"]:::neckStyle
        Concat3["Concatenate (P3 + P4_Up)<br/>(256, 9, 16)"]:::neckStyle
        Smooth3["Smooth Conv 3x3<br/>(128, 9, 16)"]:::neckStyle

        Lat3["Lateral Conv 1x1<br/>(64, 9, 16)"]:::neckStyle
        P3_Up["Upsample Bilinear<br/>to (64, 18, 32)"]:::neckStyle
        Concat2["Concatenate (P2 + P3_Up)<br/>(128, 18, 32)"]:::neckStyle
        Smooth2["Smooth Conv 3x3<br/>(64, 18, 32)"]:::neckStyle
    end

    %% Proto/Seg Head Subgraph
    subgraph SegHead ["Segmentation Proto/Mask Head"]
        UpSeg1["Upsample Stage 1<br/>(64, 18, 32)"]:::headStyle
        SkipFuse["Skip Add (UpSeg1 + Smooth2)<br/>(64, 18, 32)"]:::headStyle
        UpSeg2["Upsample Stage 2<br/>(32, 36, 64)"]:::headStyle
        ConvFinal["Final Mask Conv 1x1<br/>(1, 36, 64) Logits"]:::headStyle
    end

    %% Output
    Output["Binary Lane Mask<br/>(1, 36, 64)"]:::outputStyle

    %% Connections
    Input --> Stem
    Stem --> Stage1
    Stage1 --> Stage2
    Stage2 --> Stage3

    Stage3 --> Lat4 --> P4_Up
    Stage2 --> Concat3
    P4_Up --> Concat3
    Concat3 --> Smooth3

    Smooth3 --> Lat3 --> P3_Up
    Stage1 --> Concat2
    P3_Up --> Concat2
    Concat2 --> Smooth2

    Smooth3 --> UpSeg1
    UpSeg1 --> SkipFuse
    Smooth2 --> SkipFuse
    SkipFuse --> UpSeg2
    UpSeg2 --> ConvFinal
    ConvFinal --> Output
```

---

## Architectural Highlights

1. **Backbone**:
   - Uses strided convolutions with residual connections (YOLO style).
   - Downsamples image from 36x64 down to 5x8 while expanding feature channels from 3 to 256.
   - Activations are `SiLU` paired with `BatchNorm2d`.

2. **FPN Neck (Feature Pyramid Network)**:
   - Fuses multi-scale feature maps from low levels (fine-grained lane boundaries) and deep levels (semantic context).
   - Top-down feature upsampling ensures precise localization for thin lane lines.

3. **Segmentation Proto Head**:
   - Progressive upsampling layers bring spatial feature resolutions back to 36x64.
   - Outputs raw binary mask logits evaluated via `BCE + Dice Loss`.
