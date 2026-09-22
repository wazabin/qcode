# Stage 2: position-independent payload.
# write(1, "unpacked: stage 2\n", 18); exit_group(7).
# Assembled with `gcc -c`, extracted as a raw blob with `objcopy -O binary`.
# Only RIP-relative addressing is used, so it runs wherever mmap places it.
	.text
	.globl	stage2_start
stage2_start:
	# write(1, msg, 18)
	movl	$1, %eax			# SYS_write
	movl	$1, %edi			# fd = 1 (stdout)
	leaq	msg(%rip), %rsi			# RIP-relative pointer to the message
	movl	$18, %edx			# length
	syscall

	# exit_group(7)
	movl	$231, %eax			# SYS_exit_group
	movl	$7, %edi			# status = 7
	syscall

msg:
	.ascii	"unpacked: stage 2\n"
	.size	stage2_start, .-stage2_start
