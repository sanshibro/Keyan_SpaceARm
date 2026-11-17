from pathlib import Path
import re
root=Path.home()/'paper_revision_20261117'
p=root/'paper/main.tex';s=p.read_text()
start=s.index(r'\subsection{Representative Passage and Rolling Updates}')
end=s.index(r'\subsection{Handoff Continuity}',start)
replacement=r'''\subsection{Representative Passage and Rolling Updates}
\begin{figure*}[!tb]
\centering
\includegraphics[width=\textwidth]{passage_mechanism_v2.pdf}
\caption{Morphology changes the geometric outcome in a recorded Multi-Passage task. (a) Side projection of the UAV path and five measured configurations; rectangles depict the body collision envelope. (b)--(d) Equal-scale gate-normal sections compare the recorded bent arm with a straight-arm counterfactual at the same UAV position and yaw. Crosses mark the counterfactual distal point. (e) Signed arm clearances at the three selected samples, computed with the frozen three-dimensional geometry; zero indicates contact and the dashed line marks the 7~cm planning margin. Samples are selected nearest each gate's world-$y$ center. Projections explain the mechanism; the audit uses the full three-dimensional boxes and sampled robot model.}
\label{fig:geometry}
\end{figure*}

Figure~\ref{fig:geometry} links the full passage to its local geometric constraints. The recorded bent configurations provide arm clearances of 21.6, 23.4, and 19.1~cm at G1--G3. Straightening the arm while retaining each measured UAV position and yaw changes these values to $-4.7$, $-3.3$, and $-4.8$~cm. Thus, the shape change alters the feasibility of the illustrated states, rather than merely changing the appearance of the path.

Increasing altitude alone also faces competing local constraints. With the original map inflation and planning margins, G1 requires a straight-arm UAV height of at least 1.465~m, whereas its upper-body constraint permits at most 1.422~m. The corresponding lower/upper height intervals are separated by 4.3, 2.3, and 2.0~cm at G1--G3. These are per-gate interval gaps for the specified local geometry, not collision depths or a restriction to constant altitude over the complete task. The counterfactual explains the demonstrated route; fixed-morphology planning with a newly optimized route is a separate comparison.

\begin{figure*}[!tb]
\centering
\includegraphics[width=\textwidth]{rolling_execution_v2.pdf}
\caption{Rolling execution reconstructed from the same recorded task. (a) Actual active-window intervals W1--W7; dotted segments indicate accepted successors awaiting activation, and orange segments denote the two 1.5~s validated C$^2$ bridges. (b) The actually issued bending command and measured bending response, rather than superseded candidate tails. (c) Post-hoc body and arm clearances, displayed over the critical 0--50~cm range; dashed horizontal lines are the original audit thresholds of 6 and 2~cm. Vertical dotted lines indicate activations; green dashed lines locate the samples used in Fig.~\ref{fig:geometry}. (d) Wall-clock computation durations and nominal reserve reconstructed from trajectory metadata, indexed by window. This reserve is not corrected for execution-phase modulation.}
\label{fig:rolling}
\end{figure*}

Figure~\ref{fig:rolling} separates planning, command issuance, and physical response. Seven accepted windows cover 124.96~s of task execution, with four direct successor handoffs and two revalidated C$^2$ bridges. The measured morphology follows the issued command while local candidates are replaced. Comparing execution with a superseded candidate tail would therefore not measure tracking error. The post-hoc sampled-model audit retains minimum body and arm clearances of 0.1889 and 0.1887~m over the recorded task.

Planning and validation require 2.06--3.88~s per window in this case, below the 15~s planning deadline. The nominal metadata reserves at successor acceptance are 13.5--23.3~s. Computation durations are plotted by window because solver wall time and execution timestamps have different time bases. Supplementary Fig.~S2 retains the additional position, bending-vector, and coupling histories, while Supplementary Fig.~S8 shows the full candidate plans, including portions superseded before execution.

'''
s=s[:start]+replacement+s[end:]
s=s.replace('A paired handoff ablation reported a 72.8\\% reduction in translational jerk energy per unit path length.','A paired comparison of the earlier and revised handoff versions reported a 72.8\\% median reduction in translational jerk energy per unit path length.')
s=s.replace('A separate 15-scenario paired ablation compared position-only switching with continuity-aware handoff. Both variants completed 15/15 tasks. The supplied paired analysis reported a 72.8\\% median reduction in $E_{j/L}$, with improvement on all 15 tasks. This comparison supports the switching mechanism: smooth local trajectories alone do not ensure smooth transitions between rolling windows.','A separate 15-scenario paired version comparison evaluated the earlier position-only switching implementation against the revised continuity-aware implementation. Both versions completed 15/15 tasks. The supplied paired analysis reported a 72.8\\% median reduction in $E_{j/L}$, with improvement on all 15 tasks. This result characterizes the combined version change; an otherwise identical single-factor handoff ablation is needed to isolate the contribution of the gate and bridge.')
s=s.replace('The handoff ablation isolates switching behavior; the main success result belongs to the complete planning pipeline because separate seed and adaptive-plane ablations were not supplied.','The paired version comparison supports improved transition behavior without isolating each implementation change. The main success result belongs to the complete planning pipeline because separate seed and adaptive-plane ablations were not supplied.')
s=s.replace('配对接管消融实验进一步表明，连续接管机制使单位路径长度平移 Jerk 能量降低了 72.8%。','早期与修订接管版本的配对比较报告了单位路径长度平移 Jerk 能量中位降低 72.8%，单因素归因仍需独立接管消融。')
p.write_text(s)
print('Updated manuscript sections, captions and version-comparison attribution.')
