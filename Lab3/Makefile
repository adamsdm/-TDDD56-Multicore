################################################################################
#                     ____  _        ____  _   _   ____                        #
#                    / ___|| | _____|  _ \| | | | |___ \                       #
#                    \___ \| |/ / _ \ |_) | | | |   __) |                      #
#                     ___) |   <  __/  __/| |_| |  / __/                       #
#                    |____/|_|\_\___|_|    \___/  |_____|                      #
#                                                                              #
#                          ~ SkePU 2 main Makefile ~                           #
################################################################################


# ---------------------------------------------------------------------------- #
# Test program names and target directories.

OUT_DIR = bin

# All SkePU 2 example programs (used for 'make all-[parallel|sequential]' and 'make clean').
PROGS = addone dotproduct average median 

PTESTS = $(addprefix $(OUT_DIR)/, $(PROGS))

include Makefile.include


# ---------------------------------------------------------------------------- #
# Make recipies begin here.

# Builds all test programs, parallel implementations, using the precompiler.
all-parallel: $(PTESTS)

# Precompiles and builds a single parallel test program.
$(OUT_DIR)/%: %.cpp
	$(DBGR) $(SKEPU) -name $*_precompiled $<  -dir $(OUT_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) -o $@ $(TARGET_FLAGS)

# Precompiles and builds a single parallel test program.
$(OUT_DIR)/median: median.cpp
	$(DBGR) $(SKEPU) -name median_precompiled $<  -dir $(OUT_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) support.cpp lodepng.cpp -o $@ $(TARGET_FLAGS)
	
# Precompiles and builds a single parallel test program.
$(OUT_DIR)/average: average.cpp
	$(DBGR) $(SKEPU) -name average_precompiled $<  -dir $(OUT_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) support.cpp lodepng.cpp -o $@ $(TARGET_FLAGS)



# Deletes all temporary files (including all precompiled sources) and binaries.
clean:
	-$(RM) $(OUT_DIR)/*_precompiled.cpp $(OUT_DIR)/*_cl_source.inl $(OUT_DIR)/*.cu $(PTESTS)
	
