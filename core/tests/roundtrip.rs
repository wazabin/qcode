//! Textual round-trip tests: for every mnemonic and construct, the printed IR
//! must re-parse to an identical printed IR — i.e. `s.parse().to_string() == s`
//! for the canonical form `s` (equivalently `e.to_string().parse() == e`).
//!
//! Each case lowers a qcode program to its canonical printed form `once`, then
//! lowers and prints that form again as `twice`, and asserts `once == twice`.
//! Because `once` is already canonical, this proves the printer's output parses
//! back to the same IR. Programs avoid `varnode` declarations, which are
//! deliberately not printed (an accepted round-trip gap), and instead feed
//! inputs as typed block parameters.
//!
//! Known round-trip gaps (a printed form the parser cannot read back), excluded
//! here by construction:
//!   - `varnode` declarations are not printed.
//!   - pointer/struct types print as `T*` / `Struct`, but a value's type
//!     annotation in the grammar (`ty`) is only `iN`/`fN`; this affects `gep`
//!     results and pointer/struct-typed block parameters.

use qcode::context::Context;
use qcode::lower::lower_str;

/// Lower `src` and return its canonical printed form.
fn canon(src: &str) -> String {
    let mut ctx = Context::new();
    lower_str(&mut ctx, src).unwrap_or_else(|e| panic!("lower failed: {e}\n--- src ---\n{src}"));
    format!("{ctx}")
}

/// Assert that the printed IR for `src` re-parses to an identical print, and
/// that the printed form contains `needle` (so the construct under test is
/// actually present and not silently dropped).
#[track_caller]
fn roundtrips(src: &str, needle: &str) {
    let once = canon(src);
    assert!(
        once.contains(needle),
        "expected `{needle}` in printed IR:\n{once}"
    );
    let twice = canon(&once);
    assert_eq!(
        once, twice,
        "round-trip mismatch (printed form does not re-parse identically)\n--- once ---\n{once}\n--- twice ---\n{twice}"
    );
}

#[test]
fn roundtrip_int_binops() {
    roundtrips(
        "
        <b @x:i32 @y:i32>
            %add = @x + @y;
            %sub = @x - @y;
            %mul = @x * @y;
            %udiv = @x / @y;
            %umod = @x % @y;
            %and = @x & @y;
            %or = @x | @y;
            %xor = @x ^ @y;
            %shl = @x << @y;
            %shr = @x >> @y;
            %sshr = @x s>> @y;
            %eq = @x == @y;
            %ne = @x != @y;
            %lt = @x < @y;
            %le = @x <= @y;
            %gt = @x > @y;
            %ge = @x >= @y;
            %slt = @x s< @y;
            %sle = @x s<= @y;
            %sgt = @x s> @y;
            %sge = @x s>= @y;
            %sdiv = @x s/ @y;
            %smod = @x s% @y;
            return @x;
        ",
        "i32 @x s>> i32 @y",
    );
}

#[test]
fn roundtrip_float_binops() {
    roundtrips(
        "
        <b @x:f64 @y:f64>
            %add = @x f+ @y;
            %sub = @x f- @y;
            %mul = @x f* @y;
            %div = @x f/ @y;
            %eq = @x f== @y;
            %ne = @x f!= @y;
            %lt = @x f< @y;
            %le = @x f<= @y;
            %gt = @x f> @y;
            %ge = @x f>= @y;
            return @x;
        ",
        "i64 @x f<= i64 @y",
    );
}

#[test]
fn roundtrip_unops() {
    roundtrips(
        "
        <b @x:i32 @f:f64>
            %bneg = ~ @x;
            %neg = - @x;
            %fneg = f- @f;
            %abs = abs(@f);
            %sqrt = sqrt(@f);
            %floor = floor(@f);
            %ceil = ceil(@f);
            %round = round(@f);
            return @x;
        ",
        "abs(i64 @f)",
    );
}

#[test]
fn roundtrip_casts() {
    roundtrips(
        "
        <b @x:i32 @f:f32>
            %z = zext(i64, @x);
            %s = sext(i64, @x);
            %t = trunc(i16, @x);
            %i2f = int2float(f64, @x);
            %f2f = float2float(f64, @f);
            return @x;
        ",
        "int2float(f64, i32 @x)",
    );
}

#[test]
fn roundtrip_range() {
    roundtrips(
        "
        <b @x:i64>
            %a = @x[0:4];
            %b = @x[4:8];
            return @x;
        ",
        "[0:4]",
    );
}

#[test]
fn roundtrip_flags() {
    roundtrips(
        "
        <b @x:i32 @y:i32 @f:f64>
            %nan = nan(@f);
            %pc = popcount(@x);
            %lz = lzcount(@x);
            %c = carry(@x, @y);
            %sc = scarry(@x, @y);
            %sb = sborrow(@x, @y);
            return @x;
        ",
        "sborrow(i32 @x, i32 @y)",
    );
}

#[test]
fn roundtrip_load_store() {
    roundtrips(
        "
        <b @p:i64 @v:i32>
            %l = load(ram:4, @p);
            store(ram:4, @p <- @v);
            return %l;
        ",
        "store(ram:4, i64 @p <- i32 @v)",
    );
}

#[test]
fn roundtrip_tuple_extract() {
    roundtrips(
        "
        <b @x:i32 @y:i32>
            %t = pack(lo=@x, hi=@y);
            %e = extract(%t.lo);
            return %e;
        ",
        "pack(lo=i32 @x, hi=i32 @y)",
    );
}

#[test]
fn roundtrip_branch_and_cbranch() {
    roundtrips(
        "
        <entry @c:i8 @p:i64>
            if @c goto <t @v=@p> else goto <f>;
        <t @v:i64>
            goto <f>;
        <f>
            return @c;
        ",
        "if i8 @c goto",
    );
}

#[test]
fn roundtrip_branchind() {
    roundtrips(
        "
        <b @p:i64>
            goto [@p];
        ",
        "goto [i64 @p]",
    );
}

#[test]
fn roundtrip_assert() {
    roundtrips(
        "
        <b @c:i8>
            assert @c;
            return @c;
        ",
        "assert i8 @c",
    );
}

#[test]
fn roundtrip_block_params_carry_types() {
    // The block header must print parameter types so a re-parse recovers sizes.
    roundtrips(
        "
        <entry @x:i64>
            goto <head @hn=@x>;
        <head @hn:i64>
            return @hn;
        ",
        "<head @hn:i64>",
    );
}

#[test]
fn roundtrip_lambda_apply_and_value_return() {
    roundtrips(
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
        "apply inc(",
    );
}

#[test]
fn roundtrip_intrinsic() {
    roundtrips(
        "
        <b @x:i32 @k:i8>
            %r = $rol(@x, @k);
            return %r;
        ",
        "$rol(",
    );
}

#[test]
fn roundtrip_map() {
    roundtrips(
        "
        lambda body:
        <b @x:i32>
            return @x;
        fn main:
        <e @a:i64>
            %r = body <$> @a;
            return %r;
        ",
        "body <$> i64 @a",
    );
}

#[test]
fn roundtrip_scan() {
    roundtrips(
        "
        lambda body:
        <b @acc:i32 @x:i32>
            return @acc;
        fn main:
        <e @init:i32 @a:i64>
            %r = scanl @body @init @a;
            return %r;
        ",
        "scanl @body i32 @init i64 @a",
    );
}

#[test]
fn roundtrip_scan_with_captures() {
    roundtrips(
        "
        lambda body:
        <b @acc:i32 @x:i32>
            return @acc;
        fn main:
        <e @init:i32 @a:i64 @c:i32>
            %r = scanl (@body @c) @init @a;
            return %r;
        ",
        "scanl (@body i32 @c) i32 @init i64 @a",
    );
}

#[test]
fn roundtrip_callind() {
    roundtrips(
        "
        <b @p:i64>
            call [@p];
        ",
        "call [i64 @p]",
    );
}

#[test]
fn roundtrip_callind_with_args() {
    roundtrips(
        "
        <b @p:i64 @x:i32 @y:i32>
            call [@p](@x, @y);
        ",
        "call [i64 @p](i32 @x, i32 @y)",
    );
}

#[test]
fn roundtrip_call_with_args() {
    // A direct call's args print as `@<param>=<value>` and must re-parse.
    roundtrips(
        "
        fn callee:
        <c @a:i32 @b:i32>
            return @a;
        fn main:
        <e @x:i32 @y:i32>
            call fn callee(@a=@x, @b=@y);
        ",
        "call fn callee(",
    );
}

#[test]
fn roundtrip_return_value_at() {
    roundtrips(
        "
        <b @v:i64 @p:i64>
            return @v at @p;
        ",
        "return i64 @v at i64 @p",
    );
}
