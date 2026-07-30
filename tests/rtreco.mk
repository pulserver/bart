
tests/test-rtreco-median: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	export BART_TOOLBOX_PATH=$(ROOTDIR)			;\
	export PATH="$(ROOTDIR):$${PATH}"			;\
	. $(ROOTDIR)/scripts/rtreco.sh				;\
	$(ROOTDIR)/bart ones 1 20 -				|\
	$(ROOTDIR)/bart noise - -				|\
	$(ROOTDIR)/bart transpose 0 10 - in			;\
	$(ROOTDIR)/bart filter -m 10 -l5 in ref			;\
	$(ROOTDIR)/bart -r in -l 1024 copy in -			|\
	filter 5 - -						|\
	$(ROOTDIR)/bart resize -f 10 16 - x			;\
	$(ROOTDIR)/bart nrmse -t0 ref x				;\
	rm *.cfl *.hdr; cd .. ; rmdir $(TESTS_TMP)
	touch $@

ifneq ($(BUILDTYPE), WASM)
TESTS_BART += tests/test-rtreco-median
endif
