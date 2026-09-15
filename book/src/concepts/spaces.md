# Memory spaces and varnodes

Machine state is not made of SSA values. A register is written many times,
and memory is addressed by computed pointers. QCode keeps all of it in
**memory spaces** — uniformly addressed byte arrays — and models every read
and write as an explicit `load` or `store`.

## Spaces

| Space | Contents |
| --- | --- |
| `ram` | The guest's memory: what the program reads, writes and executes. |
| `register` | The register file of the architecture, as SLEIGH defines it: `RAX` is bytes `0..8` at some offset, `EAX` its low half, `AL` its low byte. |
| `$tempN` | A scratch space local to one lifted instruction, holding the p-code temporaries SLEIGH introduces. |
| a varnode's own space | A hand-declared `varnode T name` is a one-location space named `name`. |

`load(space:N, ptr)` reads `N` bytes at `ptr` in `space`; `store(space:N, ptr
<- v)` writes them. The address is an ordinary value, so a register access
and a memory access look alike:

```qcode,ignore
i64 %rax = load(register:8, i64 RAX);
i64 %m = load(ram:8, i64 %rax);
store(register:8, i64 RAX <- i64 %m);
```

Here `RAX` is a varnode atom: the *address* of the register inside the
`register` space. The width of the access is on the instruction, not on the
location, which is how `AL`, `AX`, `EAX` and `RAX` alias the same bytes.

## Varnodes

A varnode is a named location: a space, an offset, and a size. It is what
SLEIGH calls the same thing. Varnodes are operands only where an address is
expected — the pointer of `load`/`store`, or explicitly taken with `&name` —
and they are never SSA values themselves: reading one is a `load`.

In hand-written IR, `varnode i64 RAX;` declares `RAX` as its own space, and
its contents are read with `load(RAX:8, &RAX)`:

```qcode
varnode i64 RAX;
varnode i64 RBX;

<entry>
    %a = load(RAX:8, &RAX);
    %b = load(RBX:8, &RBX);
    %sum = %a + %b;
    store(RAX:8, &RAX <- %sum);
    goto <0x1010>;
<0x1010>
    return at %sum;
```

Lifted code uses the architecture's shared `register` space instead, with the
register names SLEIGH gives it.

## What the lifter produces

The [playground](../playground.md) shows raw lifted IR. For `add rax, rbx`
(`48 01 d8`) it reads the two registers, computes the sum, stores it, then
computes every flag from the stored value — `CF` from `carry`, `OF` from
`scarry`, `SF` from `s< 0`, `ZF` from `== 0`, `PF` from `popcount` of the low
byte — each through a `$temp` space, because that is what the p-code says.
Every `load` after a `store` to the same location is redundant, and the
first cleanup pass removes them; but the raw form is the faithful one, and it
is what the emulator checks against hardware.

## Aliasing and order

Loads and stores to the same space are ordered as written. Two spaces never
alias: a `store` to `ram` cannot change a `register` load, and a `$temp`
space dies with its instruction. Inside one space, whether two accesses alias
is a question about their addresses, which is what alias analysis in the
passes answers; the IR itself only promises program order.
