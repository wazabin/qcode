# Text format

Every QCode module prints as text and every printed module parses back, which
is how the test suite, the `qcode!` macro and this reference all write IR. The
[language reference](../langref.md) gives the form of each statement; this
page gives the shape of a program around them.

## Programs

A program is either a bare list of statements (one anonymous function) or a
list of function declarations, optionally preceded by struct types and global
varnodes:

```qcode
type node { value: 8, next: node* }

varnode i64 RAX;
varnode i64 RSP;

fn main:
<entry>
    %v = load(RAX:8, &RAX);
    %w = %v + 1;
    store(RSP:8, &RSP <- %w);
    goto <0x1010>;
<0x1010>
    return at %v;

lambda inc:
<entry @x:i64>
    %r = @x + 1;
    return %r;
```

- `type name { field: size, … }` declares a nominal struct; a field's type is
  a byte size or a pointer to another struct (`node*`).
- `varnode T name` declares a location: a register, addressed as `&name` in
  its own space `name` (see [memory spaces](spaces.md)).
- `fn name:` declares a machine function; `lambda name:` a pure value-level
  function (see [functions](functions.md)).

Statements are separated by `;`. `#` starts a comment to the end of the line;
`// -> <a>, <b>` after a terminator is not a comment but an *edge hint*, the
successors of a `call` or an indirect `goto` that the instruction itself does
not name.

## Blocks

A block is a label followed by its statements:

```qcode,ignore
<head @n:i64 @acc:i64>
    bool %done = i64 @n == i64 0x0;
    if bool %done goto <exit @r=i64 @acc> else goto <body @n=i64 @n @acc=i64 @acc>;
```

Labels are `<name>` or `<address>` (`<0x1010>`, a block at a machine
address). Parameters are declared with a type, `@n:i64`, and passed by name at
every branch to the block, `<exit @r=i64 @acc>`. See
[block arguments](block-arguments.md).

## Statements and operands

A value-producing statement is `T %name = expression;`. The type prefix is
optional on input when the expression's type is clear; the printer always
writes it. A terminator (`goto`, `if`, `switch`, `call`, `tailcall`, `return`,
`badinsn`) binds nothing and ends the block.

Operands are `T atom`, where the atom is an SSA value `%v`, a block parameter
`@p`, a varnode `RAX`, a literal `0x10` / `true` / `false`, or the address of a
varnode `&RAX`. Inside the `qcode!` macro, `{name}` captures a Rust value.

## Grammar

The parser's grammar, as compiled by `wazabin-qcode-parser`:

```pest
{{#include ../../../parser/src/qcode.pest}}
```
