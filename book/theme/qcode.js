// QCode book additions: highlight ```qcode blocks and drive the playground.
(function () {
  "use strict";

  var KEYWORDS =
    "goto if else switch default call tailcall fn lambda return at apply " +
    "load store zext sext trunc int2float float2float extract gep pack " +
    "scanl badinsn assert type varnode local abs sqrt floor ceil round " +
    "nan popcount lzcount carry scarry sborrow";

  function qcodeLanguage(hljs) {
    return {
      name: "qcode",
      keywords: { keyword: KEYWORDS, literal: "true false" },
      contains: [
        hljs.COMMENT("#", "$"),
        { className: "comment", begin: "// ->", end: "$" },
        { className: "qcode-label", begin: "<(?=[A-Za-z0-9_])", end: ">", contains: [
          { className: "qcode-param", begin: "@[A-Za-z_][A-Za-z0-9_]*" },
          { className: "qcode-value", begin: "%[A-Za-z_][A-Za-z0-9_]*" },
          { className: "qcode-type", begin: "\\b(i|f)[0-9]+\\b|\\bbool\\b" },
          { className: "number", begin: "\\b0x[0-9a-fA-F]+\\b|\\b[0-9]+\\b" }
        ] },
        { className: "qcode-value", begin: "%[A-Za-z_][A-Za-z0-9_]*" },
        { className: "qcode-param", begin: "@[A-Za-z_][A-Za-z0-9_]*" },
        { className: "qcode-intrinsic", begin: "\\$[A-Za-z_][A-Za-z0-9_]*" },
        { className: "qcode-op", begin: "(?<= )(?:s?(?:<<|>>|<=|==|!=|[<>+\\-^&|*\\/%])|f(?:==|!=|<=|<|\\+|-|\\*|\\/)|~|<\\$>)(?= )" },
        { className: "qcode-type", begin: "\\b(i|f)[0-9]+\\b|\\bbool\\b|\\b[A-Za-z_][A-Za-z0-9_]*\\*" },
        { className: "number", begin: "\\b0x[0-9a-fA-F]+\\b|\\b[0-9]+\\b" }
      ]
    };
  }

  // mdBook bundles highlight.js 10.1, which has `highlightBlock` (not the
  // later `highlightElement`) and has already run over every block by the
  // time this script loads; re-running it on the qcode blocks is harmless.
  function highlightBlock(el) {
    (hljs.highlightElement || hljs.highlightBlock).call(hljs, el);
  }

  function highlightQcode() {
    if (typeof hljs === "undefined") return;
    if (!hljs.getLanguage("qcode")) hljs.registerLanguage("qcode", qcodeLanguage);
    document.querySelectorAll("code.language-qcode").forEach(function (el) {
      if (el.textContent.trim() === "") return;
      el.removeAttribute("data-highlighted");
      highlightBlock(el);
    });
  }
  window.qcodeHighlight = highlightBlock;

  // Reference anchors: the langref headings are the Rust names, so the
  // textual keyword or operator has to be mapped onto them.
  var LANGREF = {
    load: "load", store: "store", goto: "branch", if: "cbranch", switch: "switch",
    call: "call", tailcall: "tailcall", apply: "apply", return: "return",
    badinsn: "badinsn", zext: "zext", sext: "sext", trunc: "floattoint",
    int2float: "inttofloat", float2float: "floattofloat", nan: "isfloatnan",
    popcount: "popcount", lzcount: "lzcount", carry: "carry", scarry: "scarry",
    sborrow: "sborrow", assert: "assert", pack: "tuple", extract: "extract",
    gep: "gep", scanl: "scan", abs: "floatabs", sqrt: "floatsqrt",
    ceil: "floatceil", floor: "floatfloor", round: "floatround",
    "$rol": "rol", "$ror": "ror"
  };
  var OPERATORS = {
    "==": "equal", "!=": "notequal", "<": "less", "s<": "sless", "<=": "lessequal",
    "s<=": "slessequal", "+": "add", "-": "sub", "^": "xor", "&": "and", "|": "or",
    "<<": "shiftleft", ">>": "shiftright", "s>>": "sshiftright", "*": "mul",
    "/": "div", "%": "rem", "s/": "sdiv", "s%": "srem", "f==": "equal-1",
    "f!=": "notequal-1", "f<": "less-1", "f<=": "lessequal-1", "f+": "add-1",
    "f-": "sub-1", "f*": "mul-1", "f/": "div-1", "<$>": "map"
  };

  var UNARY = { "-": "intnegate", "~": "intnot", "f-": "floatnegate" };

  // Wraps the highlighted keywords, intrinsics and operators of a code
  // element in links to the reference.
  function linkReference(code, base) {
    var spans = code.querySelectorAll(".hljs-keyword, .hljs-qcode-intrinsic, .hljs-qcode-op");
    spans.forEach(function (span) {
      var tok = span.textContent;
      var before = span.previousSibling && span.previousSibling.nodeType === 3 ? span.previousSibling.textContent : "";
      var anchor = null;
      if (span.classList.contains("hljs-qcode-op")) {
        anchor = /= $/.test(before) ? UNARY[tok] : OPERATORS[tok];
      } else {
        anchor = LANGREF[tok];
        // A keyword is only an instruction in statement position, not a
        // space or register name inside parentheses (`load(register:8, …)`).
        if (/[(,:]\s*$/.test(before)) anchor = null;
      }
      if (!anchor) return;
      var a = document.createElement("a");
      a.href = base + "#" + anchor;
      span.parentNode.insertBefore(a, span);
      a.appendChild(span);
    });
  }

  var PRESETS = [
    ["mov rax, rbx", "48 89 d8"],
    ["add rax, rbx", "48 01 d8"],
    ["xor eax, eax", "31 c0"],
    ["push rbp", "55"],
    ["mov rbp, rsp", "48 89 e5"],
    ["lea rax, [rdi+rsi*4]", "48 8d 04 b7"],
    ["cmp rdi, rsi; jl +5", "48 39 f7 7c 05"],
    ["shl rax, cl", "48 d3 e0"],
    ["rol eax, 5", "c1 c0 05"],
    ["imul rax, rdi", "48 0f af c7"],
    ["call rax", "ff d0"],
    ["ret", "c3"],
    ["movsd xmm0, [rdi]", "f2 0f 10 07"],
    ["prologue", "55 48 89 e5 48 83 ec 10"],
  ];

  window.qcodePlayground = function (init, lift) {
    var hex = document.getElementById("pg-hex");
    var address = document.getElementById("pg-address");
    var presets = document.getElementById("pg-presets");
    var status = document.getElementById("pg-status");
    var asm = document.getElementById("pg-asm");
    var qcode = document.getElementById("pg-qcode");
    var ready = false;
    var timer = null;

    PRESETS.forEach(function (p) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = p[0];
      b.title = p[1];
      b.addEventListener("click", function () { hex.value = p[1]; run(); });
      presets.appendChild(b);
    });

    function run() {
      if (!ready) return;
      var bytes = hex.value.trim();
      if (bytes === "") {
        status.textContent = "Type x86-64 bytes as hex, or pick an instruction.";
        status.className = "playground-status";
        asm.textContent = "";
        qcode.textContent = "";
        return;
      }
      var addr;
      try { addr = BigInt(address.value.trim() || "0x1000"); } catch (e) { addr = 0x1000n; }
      var t0 = performance.now();
      var result;
      try { result = JSON.parse(lift(bytes, addr)); } catch (e) { result = { error: String(e) }; }
      var ms = (performance.now() - t0).toFixed(1);
      if (result.error) {
        status.textContent = result.error;
        status.className = "playground-status error";
        return;
      }
      status.textContent = result.instructions.length + " instruction" +
        (result.instructions.length === 1 ? "" : "s") + " · " + ms + " ms";
      status.className = "playground-status";
      asm.textContent = result.instructions.map(function (i) {
        return i.address + "  " + i.bytes.padEnd(16) + " " + i.text;
      }).join("\n");
      qcode.textContent = result.qcode;
      highlightBlock(qcode);
      linkReference(qcode, "langref.html");
    }

    hex.addEventListener("input", function () { clearTimeout(timer); timer = setTimeout(run, 150); });
    address.addEventListener("input", function () { clearTimeout(timer); timer = setTimeout(run, 150); });

    init().then(function () {
      ready = true;
      status.textContent = "Type x86-64 bytes as hex, or pick an instruction.";
      var initial = new URLSearchParams(location.search).get("hex");
      if (initial) hex.value = initial;
      run();
    }).catch(function (e) {
      status.textContent = "The lifter failed to load: " + e;
      status.className = "playground-status error";
    });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", highlightQcode);
  } else {
    highlightQcode();
  }
})();
