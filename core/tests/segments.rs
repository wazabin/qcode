//! `to_segments` fidelity: concatenating an instruction's rendered token
//! segments must reproduce its canonical `Display` byte-for-byte. This pins the
//! textual format as the single source of truth — the colored/clickable segment
//! view (used by the GUI) can never drift from the parser's canonical form.
//!
//! Every grammar-constructible mnemonic is exercised below. `scan` and
//! `pcode_op` have no parser production (the same gap the round-trip tests note),
//! so they are not reachable through `lower_str` and are covered only by the
//! mirrored logic in `instruction_segments`.

use qcode::context::Context;
use qcode::lower::lower_str;
use qcode::value::insn::segment::instruction_segments;

/// Lower `src`, then assert that for every instruction in every block the
/// concatenated segment text equals the instruction's `Display`.
#[track_caller]
fn check(src: &str) {
    let mut ctx = Context::new();
    lower_str(&mut ctx, src).unwrap_or_else(|e| panic!("lower failed: {e}\n--- src ---\n{src}"));

    let mut checked = 0usize;
    for block in ctx.blocks() {
        for insn in block.iter() {
            let concat: String = instruction_segments(&insn)
                .iter()
                .map(|t| t.text.as_str())
                .collect();
            assert_eq!(
                concat,
                insn.as_statement().to_string(),
                "segment concat must equal Display"
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "no instructions checked for:\n{src}");
}

#[test]
fn int_binops() {
    check(
        "
        <b @x:i32 @y:i32>
            %add = @x + @y;
            %sub = @x - @y;
            %mul = @x * @y;
            %and = @x & @y;
            %or = @x | @y;
            %xor = @x ^ @y;
            %shl = @x << @y;
            %sshr = @x s>> @y;
            %eq = @x == @y;
            %slt = @x s< @y;
            %sdiv = @x s/ @y;
            return @x;
        ",
    );
}

#[test]
fn bool_and_float_binops() {
    // Comparison results are `bool`, `bool false` renders as `false`, and the
    // negation `x == false` round-trips through segments.
    check(
        "
        <b @x:i8 @y:i8>
            %cmp = @x == @y;
            %neg = %cmp == false;
            return @x;
        ",
    );
    check(
        "
        <b @x:f64 @y:f64>
            %add = @x f+ @y;
            %div = @x f/ @y;
            %le = @x f<= @y;
            return @x;
        ",
    );
}

#[test]
fn unops() {
    check(
        "
        <b @x:i32 @f:f64>
            %neg = - @x;
            %not = ~ @x;
            %abs = abs(@f);
            %sqrt = sqrt(@f);
            return @x;
        ",
    );
}

#[test]
fn casts_and_range() {
    check(
        "
        <b @x:i32 @f:f32>
            %z = zext(i64, @x);
            %s = sext(i64, @x);
            %i2f = int2float(f64, @x);
            %f2f = float2float(f64, @f);
            %f2i = trunc(i16, @f);
            return @x;
        ",
    );
    check(
        "
        <b @x:i64>
            %a = @x[0:4];
            %b = @x[4:8];
            return @x;
        ",
    );
}

#[test]
fn flags() {
    check(
        "
        <b @x:i32 @y:i32 @f:f64>
            %c = carry(@x, @y);
            %sc = scarry(@x, @y);
            %sb = sborrow(@x, @y);
            %pc = popcount(@x);
            %lz = lzcount(@x);
            %nan = nan(@f);
            return @x;
        ",
    );
}

#[test]
fn memory() {
    check(
        "
        <b @p:i64 @v:i32>
            %l = load(ram:4, @p);
            store(ram:4, @p <- @v);
            return @p;
        ",
    );
}

#[test]
fn tuple_and_extract() {
    check(
        "
        <b @x:i32 @y:i64>
            %t = pack(lo=@x, hi=@y);
            %e = extract(%t.lo);
            return @x;
        ",
    );
}

#[test]
fn control_flow() {
    check(
        "
        <entry @c:i8 @p:i64>
            if @c goto <t @v=@p> else goto <f>;
        <t @v:i64>
            goto <f>;
        <f>
            return @c;
        ",
    );
    check(
        "
        <b @p:i64>
            goto [@p];
        ",
    );
    check(
        "
        <b @v:i64 @p:i64>
            return @v at @p;
        ",
    );
}

/// A resolved jump table: arms carry block arguments like any other branch
/// target, and the default is optional — a table behind a bounds check is total
/// over the values it lists.
#[test]
fn switch_terminator() {
    check(
        "
        <entry @i:i64 @p:i64>
            switch @i { 0x0 => <a>, 0x3 => <b @v=@p>, default => <d> };
        <a>
            return @i;
        <b @v:i64>
            return @v;
        <d>
            return @i;
        ",
    );
    check(
        "
        <entry @i:i64>
            switch @i { 0x0 => <a>, 0x1 => <b> };
        <a>
            return @i;
        <b>
            return @i;
        ",
    );
}

#[test]
fn assert_stmt() {
    check(
        "
        <b @c:i8>
            assert @c;
            return @c;
        ",
    );
}

#[test]
fn intrinsic() {
    check(
        "
        <b @x:i32 @k:i8>
            %r = $rol(@x, @k);
            return @x;
        ",
    );
}

#[test]
fn apply_and_map() {
    check(
        "
        lambda inc:
        <b @x:i64>
            %r = @x + 1;
            return %r;
        fn main:
        <e @y:i64>
            %v = apply inc(@y);
            return %v;
        ",
    );
    check(
        "
        lambda body:
        <b @x:i32>
            return @x;
        fn main:
        <e @a:i64>
            %r = body <$> @a;
            return @a;
        ",
    );
}

#[test]
fn calls() {
    check(
        "
        fn callee:
        <c @a:i32 @b:i32>
            return @a;
        fn main:
        <e @x:i32 @y:i32>
            call fn callee(@a=@x, @b=@y);
        ",
    );
    check(
        "
        <b @p:i64 @x:i32 @y:i32>
            call [@p](@x, @y);
        ",
    );
}
