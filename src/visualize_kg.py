"""
MindPulse — Module 3  |  Knowledge Graph Visualizer  (v2 — fixed layout)
"""
import os, sys, math
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from knowledge_graph import build_knowledge_graph

os.makedirs("outputs", exist_ok=True)

NODE_COLORS = {
    "StressState":    "#E74C3C",
    "TriggerContext": "#3498DB",
    "LocationContext":"#27AE60",
    "SocialContext":  "#9B59B6",
    "GestureProfile": "#F39C12",
    "Intervention":   "#1ABC9C",
}
NODE_SIZES_PX = {
    "StressState":38,"TriggerContext":28,"LocationContext":28,
    "SocialContext":28,"GestureProfile":28,"Intervention":20,
}
EDGE_COLORS = {
    "TIER_APPROPRIATENESS":      "#E74C3C",
    "TRIGGER_AFFINITY":          "#3498DB",
    "LOCATION_FEASIBILITY":      "#27AE60",
    "SOCIAL_FEASIBILITY":        "#9B59B6",
    "GESTURE_URGENCY":           "#F39C12",
    "TIER_TRIGGER_COOCCURRENCE": "#95A5A6",
}

def short_label(node_id):
    parts = node_id.split(":")
    if len(parts)==2:
        tag,val=parts
        return val if tag=="Intervention" else val.replace("_","\n")
    return node_id

def compute_positions(G, canvas=900):
    cx,cy=canvas/2,canvas/2
    pos={}
    def ring(nodes,radius,angle_offset=0):
        n=len(nodes)
        for i,node in enumerate(nodes):
            angle=2*math.pi*i/n+angle_offset
            pos[node]=(cx+radius*math.cos(angle),cy+radius*math.sin(angle))
    by_type={t:[] for t in NODE_COLORS}
    for node,data in G.nodes(data=True):
        by_type[data.get("node_type","Intervention")].append(node)
    for t in by_type: by_type[t].sort()
    ring(by_type["StressState"],    radius=80,  angle_offset=-math.pi/4)
    ring(by_type["GestureProfile"], radius=185, angle_offset=math.pi/8)
    ring(by_type["TriggerContext"], radius=285, angle_offset=-math.pi/6)
    ring(by_type["LocationContext"],radius=375, angle_offset=math.pi*0.6)
    ring(by_type["SocialContext"],  radius=375, angle_offset=-math.pi*0.4)
    ring(by_type["Intervention"],   radius=490, angle_offset=math.pi/22)
    return pos

def build_interactive_html(G, output_path="outputs/kg_interactive.html", weight_threshold=0.60):
    try:
        from pyvis.network import Network
    except ImportError:
        print("[!] pyvis not installed."); return

    net=Network(height="900px",width="100%",bgcolor="#1a1a2e",
                font_color="#FFFFFF",directed=True,notebook=False)
    net.toggle_physics(False)
    canvas=900
    pos=compute_positions(G,canvas=canvas)

    for node_id,data in G.nodes(data=True):
        ntype=data.get("node_type","Unknown")
        color=NODE_COLORS.get(ntype,"#BDC3C7")
        size=NODE_SIZES_PX.get(ntype,20)
        label=short_label(node_id)
        x,y=pos.get(node_id,(canvas/2,canvas/2))
        tip=[f"<b>{node_id}</b>",f"Type: {ntype}"]
        if ntype=="Intervention":
            tip+=[f"Name: {data.get('intervention_name','')}",
                  f"Category: {data.get('intervention_type','')}",
                  f"Duration: {data.get('duration','')} min",
                  f"Target tiers: {', '.join(data.get('target_tiers',[]))}",
                  f"Excludes social: {data.get('excludes_social',[])}",
                  f"Excludes location: {data.get('excludes_location',[])}"]
        elif ntype=="StressState":
            tip.append(f"Tier index: {data.get('tier_index','')}")
        net.add_node(node_id,label=label,
                     color={"background":color,"border":"#FFFFFF",
                            "highlight":{"background":"#FFD700","border":"#FFF"}},
                     size=size,title="<br>".join(tip),x=x,y=y,
                     font={"size":11,"color":"#FFFFFF","bold":True},borderWidth=1.5)

    shown=0
    for u,v,data in G.edges(data=True):
        w=data.get("weight",0.0); etype=data.get("edge_type","")
        if w<weight_threshold: continue
        color=EDGE_COLORS.get(etype,"#7F8C8D")
        width=1.2 if w<0.80 else 3.0
        net.add_edge(u,v,color={"color":color,"opacity":0.70},width=width,
                     title=f"{etype}<br>weight: {w:.2f}",
                     arrows={"to":{"enabled":True,"scaleFactor":0.45}},
                     smooth={"type":"curvedCW","roundness":0.12})
        shown+=1

    node_legend_rows="".join(
        f'<span style="color:{c};">&#9679;</span> {t}&nbsp;&nbsp;'
        for t,c in NODE_COLORS.items())
    edge_legend_rows="".join(
        f'<span style="color:{c};">&#9644;</span> {t.replace("_"," ").title()}<br>'
        for t,c in EDGE_COLORS.items())
    legend=f"""
    <div style="position:fixed;top:16px;left:16px;background:rgba(15,15,30,0.93);
                padding:14px 18px;border-radius:10px;font-family:monospace;
                font-size:12px;color:#EEE;z-index:9999;border:1px solid #444;min-width:280px;">
      <b style="font-size:13px;">MindPulse — Module 3 KG</b>
      <hr style="border-color:#444;margin:8px 0;">
      <b>Node Types</b><br><div style="line-height:2;">{node_legend_rows}</div>
      <hr style="border-color:#444;margin:8px 0;">
      <b>Edge Types&nbsp;<span style="font-weight:normal;color:#AAA;">(weight &ge; {weight_threshold})</span></b><br>
      {edge_legend_rows}
      <hr style="border-color:#444;margin:8px 0;">
      <span style="color:#AAA;font-size:11px;">
        {G.number_of_nodes()} nodes &nbsp;|&nbsp; {shown} edges shown<br>
        Scroll to zoom &bull; Drag canvas to pan &bull; Hover for details
      </span>
    </div>"""

    html=net.generate_html()
    html=html.replace("</body>",legend+"\n</body>")
    with open(output_path,"w",encoding="utf-8") as f: f.write(html)
    print(f"[✓] Interactive HTML → {output_path}  ({G.number_of_nodes()} nodes | {shown} edges)")

def build_static_overview(G, output_path="outputs/kg_overview.png"):
    WEIGHT_THRESHOLD=0.75
    fig,ax=plt.subplots(figsize=(22,16))
    fig.patch.set_facecolor("#0F0F1A"); ax.set_facecolor("#0F0F1A")
    raw_pos=compute_positions(G,canvas=1000)
    pos={n:(x/1000*2-1,-(y/1000*2-1)) for n,(x,y) in raw_pos.items()}
    H=nx.DiGraph(); H.add_nodes_from(G.nodes(data=True))
    for u,v,d in G.edges(data=True):
        if d.get("weight",0)>=WEIGHT_THRESHOLD: H.add_edge(u,v,**d)
    for etype,color in EDGE_COLORS.items():
        elist=[(u,v) for u,v,d in H.edges(data=True) if d.get("edge_type")==etype]
        if not elist: continue
        widths=[max(0.5,H[u][v].get("weight",0.5)*2.2) for u,v in elist]
        nx.draw_networkx_edges(H,pos,edgelist=elist,ax=ax,edge_color=color,
                               alpha=0.55,width=widths,arrows=True,arrowsize=9,
                               arrowstyle="-|>",connectionstyle="arc3,rad=0.10")
    for ntype,color in NODE_COLORS.items():
        nlist=[n for n,d in G.nodes(data=True) if d.get("node_type")==ntype]
        sz={"StressState":900,"TriggerContext":600,"LocationContext":550,
            "SocialContext":550,"GestureProfile":600,"Intervention":380}.get(ntype,400)
        nx.draw_networkx_nodes(G,pos,nodelist=nlist,ax=ax,node_color=color,
                               node_size=sz,alpha=0.95,linewidths=1.5,edgecolors="#FFF")
    labels={n:short_label(n) for n in G.nodes()}
    nx.draw_networkx_labels(G,pos,labels=labels,ax=ax,font_size=6,
                            font_color="#FFF",font_weight="bold")
    node_leg=[mpatches.Patch(facecolor=c,edgecolor="#FFF",lw=0.8,label=t)
              for t,c in NODE_COLORS.items()]
    edge_leg=[Line2D([0],[0],color=c,lw=2,label=t.replace("_"," ").title())
              for t,c in EDGE_COLORS.items()]
    l1=ax.legend(handles=node_leg,loc="upper left",title="Node Types",title_fontsize=9,
                 fontsize=8,framealpha=0.85,facecolor="#1a1a2e",labelcolor="#EEE",edgecolor="#555")
    l1.get_title().set_color("#FFF"); ax.add_artist(l1)
    l2=ax.legend(handles=edge_leg,loc="upper right",
                 title=f"Edge Types (weight ≥ {WEIGHT_THRESHOLD})",title_fontsize=9,
                 fontsize=8,framealpha=0.85,facecolor="#1a1a2e",labelcolor="#EEE",edgecolor="#555")
    l2.get_title().set_color("#FFF")
    ax.set_title(f"MindPulse Module 3 — Knowledge Graph\n"
                 f"{G.number_of_nodes()} nodes  ·  {G.number_of_edges()} total edges  ·  "
                 f"{H.number_of_edges()} shown (weight ≥ {WEIGHT_THRESHOLD})",
                 fontsize=15,color="#FFF",pad=18,fontweight="bold")
    counts="  ".join(f"{t}: {sum(1 for _,d in G.nodes(data=True) if d.get('node_type')==t)}"
                     for t in NODE_COLORS)
    ax.text(0.5,-0.01,counts,transform=ax.transAxes,ha="center",va="top",fontsize=8,
            color="#AAA",bbox=dict(boxstyle="round,pad=0.4",facecolor="#1a1a2e",edgecolor="#555"))
    ax.axis("off"); plt.tight_layout(pad=1.5)
    plt.savefig(output_path,dpi=160,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Static PNG  → {output_path}")

if __name__=="__main__":
    print("Building Knowledge Graph...")
    G=build_knowledge_graph()
    print(f"[✓] KG ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
    build_interactive_html(G)
    build_static_overview(G)
    print("\n[✓] Done. Open outputs/kg_interactive.html in Chrome or Edge.\n")
