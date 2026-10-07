# Retained rejected contact inputs at 204 Myr

Captured read-only on 2026-09-22 from production run `production-pr164-168`,
source `3d0256419a0131e3f003473ee701718d51f25fa7`. The user authorized investigating
the long 204-to-206 Myr step. Inputs came from completed rejected `deform` frames
retained by the live adaptive-retry exceptions, with repeated array reads checked
for equality. They are frozen solver inputs, not checkpoints.

Files depth-2 through depth-7 represent contact region 9 at successively shorter intervals from
0.5 to 0.015625 Myr. Four of five vertices are rigid; three faces share the one
free vertex. Units and variable names match `contact_response.redistribute`.
Run with radius 6371 km, smoothing length 100 km, 1024 iterations, tolerance
1e-8, maximum speed 100 km/Myr, and the saved viscosity weights.

The parent source rejects all six with `Dependent active area constraints have
inconsistent linearized requirements`. Its trial working set contains three
upper bounds, but the least-squares reaction has a negative multiplier. Testing
equality consistency before removing that bound incorrectly aborts the existing
inequality working-set correction. The repaired order returns one binding face
and passes the original nonlinear bounds and force checks in two SQP iterations.

Depth-0 and depth-1 capture the rejected 2 and 1 Myr region-1 solves (2473 and
2458 faces). The original line search fails after evaluating an inverted face
with positive area. Keeping iterates on the outward-oriented geometry branch
solves these inputs in eight and four iterations, including the caller's
orientation and face-quality checks. See CONTACT-204-INVESTIGATION.md.
