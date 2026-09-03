/* Board support for the wazabin VM.
 *
 * The harness measures wall time on the host and reads the verification result
 * out of a register, so the triggers have nothing to do here. They are kept as
 * real (non-inlined) functions so a profile can still see the benchmark's
 * boundaries.
 *
 * The freestanding builds have no libc, but GCC still lowers struct copies and
 * array initialisation to calls to these four, so they have to exist.
 */
#include "support.h"

void __attribute__ ((noinline)) initialise_board (void) { }
void __attribute__ ((noinline)) start_trigger (void) { }
void __attribute__ ((noinline)) stop_trigger (void) { }

void *
memcpy (void *dst, const void *src, unsigned long n)
{
  unsigned char *d = dst;
  const unsigned char *s = src;
  while (n--) *d++ = *s++;
  return dst;
}

void *
memset (void *dst, int c, unsigned long n)
{
  unsigned char *d = dst;
  while (n--) *d++ = (unsigned char) c;
  return dst;
}

void *
memmove (void *dst, const void *src, unsigned long n)
{
  unsigned char *d = dst;
  const unsigned char *s = src;
  if (d < s)
    while (n--) *d++ = *s++;
  else
    {
      d += n; s += n;
      while (n--) *--d = *--s;
    }
  return dst;
}

unsigned long
strlen (const char *s)
{
  const char *p = s;
  while (*p) p++;
  return (unsigned long) (p - s);
}

char *
strchr (const char *s, int c)
{
  for (;; s++)
    {
      if (*s == (char) c) return (char *) s;
      if (!*s) return 0;
    }
}

int
memcmp (const void *a, const void *b, unsigned long n)
{
  const unsigned char *x = a, *y = b;
  while (n--)
    {
      if (*x != *y) return (int) *x - (int) *y;
      x++; y++;
    }
  return 0;
}
