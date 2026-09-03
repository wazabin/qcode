/* Embench build configuration for the wazabin VM "board".
   No chip support file, and cache warming is pointless on an emulator. */
#ifndef CONFIG_H
#define CONFIG_H
#define HAVE_BOARDSUPPORT_H 1
/* WARMUP_HEAT and GLOBAL_SCALE_FACTOR come from the command line, as they do
   in Embench's own build. */
#define CPU_MHZ 1
#endif
