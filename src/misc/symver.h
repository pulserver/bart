

// For binary compatibility with older Linux distributions
// downgrade the version of these symbols.

#define SYMVER(p, x, v) asm (".symver " #p #x ", " #x "@GLIBC_" #v)
SYMVER(, fmod, 2.2.5);
SYMVER(, fmodf, 2.2.5);
SYMVER(__isoc23_, strtol, 2.2.5);
SYMVER(__isoc23_, strtoll, 2.2.5);
SYMVER(__isoc23_, strtoul, 2.2.5);
SYMVER(__isoc23_, strtoull, 2.2.5);
SYMVER(__isoc23_, sscanf, 2.2.5);
#undef SYMVER


