# x86-64 to QCode

What does an instruction look like in QCode? Type its bytes, or pick one.
The lifter runs in your browser — the same `wazabin-qcode-sleigh` code path
as `qcode-lift --arch x64`, without cleanup passes, so this is the raw IR
exactly as the emulator sees it.

<div class="playground" id="playground">
  <div class="playground-input">
    <label>Bytes <input type="text" id="pg-hex" placeholder="48 01 d8" spellcheck="false" autocomplete="off"></label>
    <label>Address <input type="text" id="pg-address" value="0x1000" size="8" spellcheck="false" autocomplete="off"></label>
  </div>
  <div class="playground-presets" id="pg-presets"></div>
  <p class="playground-status" id="pg-status">Loading the lifter…</p>
  <pre class="playground-asm" id="pg-asm"></pre>
  <pre class="playground-qcode"><code class="language-qcode" id="pg-qcode"></code></pre>
</div>

<script type="module">
import init, { lift } from "./pkg/qcode_web.js";
window.qcodePlayground(init, lift);
</script>

Mnemonics in the output link to their entry in the
[language reference](langref.md). Several instructions may be given at once;
lifting stops at the first byte sequence that does not decode.
