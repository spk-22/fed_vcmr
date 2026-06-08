"""
Generate FedVCMR Architecture as PNG using Graphviz (DOT format)

Graphviz is lighter than LaTeX and produces clean ML pipeline diagrams.

Install: pip install graphviz

Run: python scripts/generate_graphviz.py
"""

import os
os.makedirs("C:/prism/outputs/figures", exist_ok=True)

# Graphviz DOT format - clean ML pipeline style
dot_code = """digraph FedVCMR {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fontname="Arial"];
    edge [fontname="Arial", fontsize=10];
    
    // Graph styling
    graph [bgcolor="#F8F9FA", fontname="Arial", fontsize=14, labelloc="t", 
           label="FedVCMR: On-Device Video Corpus Moment Retrieval"];
    
    // Colors
    node [fillcolor="#D6EAF8", color="#2471A3", penwidth=2, fontsize=11];
    
    // TRACK A: Text Encoder
    subgraph cluster_a {
        style=filled;
        fillcolor="#D6EAF8";
        color="#2471A3";
        penwidth=2;
        label="① Text Encoder (MobileCLIP-S1)";
        
        query [label="Text Query"];
        tokenizer [label="Tokenizer\\n(77 tokens)"];
        trans [label="6× Transformer\\nLayers"];
        proj [label="Linear Proj\\n+ L2 Norm"];
        q [label="q ∈ ℝ⁵¹²", fillcolor="#A9CCE3"];
        
        query -> tokenizer -> trans -> proj -> q;
    }
    
    // TRACK B: Video Index
    subgraph cluster_b {
        style=filled;
        fillcolor="#FAD7A0";
        color="#CA6F1E";
        penwidth=2;
        label="② Video Feature Index (mmap, 12 MB)";
        node [fillcolor="#FAD7A0", color="#CA6F1E"];
        
        F [label="Features Blob\\n819 videos × 16 frames\\n× 512-dim"];
    }
    
    // PIPELINE: 5 Stages
    subgraph cluster_pipeline {
        style=filled;
        fillcolor="#D5F5E3";
        color="#1E8449";
        penwidth=2;
        label="DGSE Scoring + Temporal Grounding Pipeline";
        
        // Stage 1
        s1 [label="① Dot Product\\ns(q,fᵢ)=q·fᵢ", fillcolor="#A9DFBF", color="#1E8449"];
        
        // Stage 2
        s2 [label="② Temporal Smoothing\\ns̃ₑ=mean(s)", fillcolor="#A9DFBF", color="#1E8449"];
        
        // Stage 3
        s3 [label="③ Hubness Suppress\\ns*=s̃-λΔμ", fillcolor="#A9DFBF", color="#1E8449"];
        
        // Stage 4
        s4 [label="④ Top-K Ranking\\nK=10", fillcolor="#A9DFBF", color="#1E8449"];
        
        // Stage 5
        s5 [label="⑤ Temporal Grounding\\nthreshold=92%", fillcolor="#D7BDE2", color="#7D3C98"];
        
        // Output
        output [label="Retrieved Moment\\n[tₛ, tₑ]", fillcolor="#EDE0F5", color="#7D3C98"];
        
        s1 -> s2 -> s3 -> s4 -> s5 -> output;
    }
    
    // Connections from tracks to pipeline
    edge [style=dashed, penwidth=1.5];
    q -> s1 [color="#2471A3", label="q"];
    F -> s1 [color="#CA6F1E", label="F"];
    
    // Stats
    stats [label="Backbone: MobileCLIP-S1 | R@1/R@5: 24.75%/52.08% | Latency: 691ms avg",
           shape=plaintext, fontsize=9, fillcolor="none", color="none"];
}
"""

OUT = "C:/prism/outputs/figures/fedvcmr_architecture.dot"
with open(OUT, "w", encoding="utf-8") as f:
    f.write(dot_code)

print(f"✓ Graphviz DOT file created: {OUT}")
print(f"\n📋 TO RENDER:")
print(f"\n1️⃣  Install Graphviz: choco install graphviz")
print(f"\n2️⃣  Python package: pip install graphviz")
print(f"\n3️⃣  Then run:")
print(f"   dot -Tpng {OUT} -o C:/prism/outputs/figures/fedvcmr_architecture_graphviz.png")
print(f"\n4️⃣  Or use online: https://dreampuf.github.io/GraphvizOnline/")
print(f"   Copy-paste the .dot content and render")
print(f"\n✓ Output will be: C:/prism/outputs/figures/fedvcmr_architecture_graphviz.png")
