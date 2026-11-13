"""Reference shaping around retargeting: the operator frame alignment at a Start/Resume
(``anchor``, before the retargeting), the convergence ramp from the robot's measured pose
(``converge``), the per-arm hold on hand-tracking loss (``arm_hold``) and the ramp
primitives they share (``ramp``). Safety-relevant: these are the only rate limits between
a Start/Resume and the robot's joint targets.
"""
