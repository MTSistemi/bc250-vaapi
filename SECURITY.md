# Security policy

## Reporting a vulnerability

Please use GitHub's **private vulnerability reporting** (Security → Report a
vulnerability) rather than opening a public issue. It keeps the report
confidential until a fix ships.

We acknowledge within a few working days. There is no bounty programme.

## What counts here

This is a VA-API driver: it is loaded **into every process that asks for
hardware video**, a browser, a media player, OBS, Steam, and it parses the
bitstreams those programs hand it. The things worth reporting:

- a crafted H.264 or HEVC stream that makes the decoder read or write out of
  bounds, crash the host process, or run code in it;
- a buffer, surface or parameter from the VA-API client that is trusted
  without a bounds check;
- the compute encoder writing outside the buffers it was given, or leaking
  GPU memory of another process into a bitstream;
- anything in the packages (`packaging/`) that installs with wider permissions
  than it needs.

## What does not count

- Bugs in Mesa, the kernel's amdgpu driver or libva itself: please report
  those upstream.
- Wrong output that is only wrong (a mismatch against the reference decoder
  with no memory-safety impact): that is a normal bug, open an issue.
- Anything in [simpmix/bc250-encoding-decoding-fix][upstream] that we have not
  changed: report it there as well, so it gets fixed for everyone.

## Supported versions

Only the latest release and `main` get fixes.

[upstream]: https://github.com/simpmix/bc250-encoding-decoding-fix
