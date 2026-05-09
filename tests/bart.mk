

tests/test-bart: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(ROOTDIR)/bart ones 2 1 1 o1.ra			;\
	cp $(ROOTDIR)/bart ./ones				;\
	./ones 2 1 1 o2.ra					;\
	$(ROOTDIR)/bart nrmse -t 0. o1.ra o2.ra			;\
	rm ones *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-bart

