# Contributing

Thanks for looking. This driver is loaded into other people's programs and
decodes streams it did not produce, so a few things matter more than usual.

## The rules that matter

- **Bit-exact or it does not go in.** The decoders are compared byte for byte
  against the reference decoder on the conformance streams. A change that
  moves a single sample needs to say why, in the pull request.
- **Every length from a bitstream or a VA-API buffer is untrusted.** Check it
  against the buffer before using it. A crash in the decoder is a crash in the
  browser or the player that loaded it.
- **Measured, not estimated.** Performance changes come with numbers from real
  hardware: the board, the stream, the frames per second before and after.

## Before a pull request

- `ctest --test-dir approach1-compute-encoder/build` passes;
- the change is tested on a BC-250 (or says clearly it was not);
- no credentials anywhere: `bash scripts/check-secrets.sh` must say so. The
  same check runs in CI on every push.

## Where changes go

Work that belongs in the shared encoder or GPU layer is best proposed to
[simpmix/bc250-encoding-decoding-fix][upstream] as well, so both trees keep
the fix.

## Licence

GPL-3.0-only, like the upstream tree. By contributing you agree your change
is released under it.

[upstream]: https://github.com/simpmix/bc250-encoding-decoding-fix
