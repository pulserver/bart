
tests/test-sort-reorder: sort index bin vec nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/vec -- 0 3 2 5 10 -3 in.ra					;\
	$(TOOLDIR)/vec -- -3 0 2 3 5 10 ref.ra					;\
	$(TOOLDIR)/sort in.ra idx.ra						;\
	$(TOOLDIR)/bin -o idx.ra in.ra sorted.ra				;\
	$(TOOLDIR)/nrmse -t 0. ref.ra sorted.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-sort-reorder

