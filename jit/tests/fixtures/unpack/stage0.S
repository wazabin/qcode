# Stage 0: the real ELF entry point (_start), living in .text.
#
# It XOR-decrypts the embedded, encrypted copy of stage 1 IN PLACE inside a
# RWX section (.rwx, flags "awx" -> SHF_ALLOC|SHF_WRITE|SHF_EXECINSTR, so the
# linker emits a PF_R|PF_W|PF_X PT_LOAD), then jumps into it.
#
# The build script defines KEY1 (the stage-1 XOR key) via -DKEY1=... and
# places the encrypted stage-1 blob in the .rwx section through .incbin.
	.text
	.globl	_start
_start:
	leaq	stage1_buf(%rip), %rdi		# pointer into the RWX buffer
	movl	$(stage1_end - stage1_buf), %ecx	# byte count (assemble-time const)
1:
	movb	(%rdi), %al
	xorb	$KEY1, %al			# decrypt one byte in place
	movb	%al, (%rdi)
	incq	%rdi
	decl	%ecx
	jnz	1b

	leaq	stage1_buf(%rip), %rax
	jmp	*%rax				# enter stage 1

# RWX buffer holding the (initially encrypted) stage-1 image, decrypted above.
	.section	.rwx, "awx", @progbits
	.align	16
	.globl	stage1_buf
stage1_buf:
	.incbin	"stage1.enc"
stage1_end:
