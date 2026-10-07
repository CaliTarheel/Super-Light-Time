# Archival canvas capture

This tool saves the app's own canvas to a Google Drive-synced folder. It names
files by geologic age (`total - frame.time_myr`) and renders at the requested
browser viewport size. It requires the selected frame's fine display grid
before it writes a PNG; a timeout or wrong frame leaves the destination alone.

```powershell
npm ci
node capture.js --out "G:\My Drive\Deep Time\frames-ma" --vw 4096 --vh 2048 --total 1000 --watch --interval 120
```

To replace one known bad image, use its zero-based frame index:

```powershell
node capture.js --out "G:\My Drive\Deep Time\frames-ma" --vw 4096 --vh 2048 --total 1000 --from 293 --to 293 --replace
```

The watcher reads the selected run from `/api/status` on each pass. A branch or
restore-point transition can change that selection without changing the output
folder. Capture only after the server has loaded the intended run. Check the
first new image visually and confirm its Drive sync before relying on the
archive.
