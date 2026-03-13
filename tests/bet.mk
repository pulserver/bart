
tests/test-bet-phantom: phantom bet threshold cabs nrmse
	set -e; mkdir $(TESTS_TMP); cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -x64 -3 phantom.ra					;\
	$(TOOLDIR)/cabs phantom.ra abs.ra					;\
	$(TOOLDIR)/threshold -B 0 abs.ra mask_ref.ra				;\
	$(TOOLDIR)/bet -b 0.1 -i 400 phantom.ra mask_bet.ra			;\
	$(TOOLDIR)/nrmse -t 0.3 mask_bet.ra mask_ref.ra				;\
	rm *.ra; cd ..; rmdir $(TESTS_TMP)
	touch $@

tests/test-bet-scale: phantom scale bet nrmse
	set -e; mkdir $(TESTS_TMP); cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -x64 -3 p.ra						;\
	$(TOOLDIR)/scale 2 p.ra p2.ra						;\
	$(TOOLDIR)/bet p.ra m1.ra						;\
	$(TOOLDIR)/bet p2.ra m2.ra						;\
	$(TOOLDIR)/nrmse -t 0.15 m1.ra m2.ra					;\
	rm *.ra; cd ..; rmdir $(TESTS_TMP)
	touch $@

tests/test-bet-deterministic: phantom bet nrmse
	set -e; mkdir $(TESTS_TMP); cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -3 -x128 p.ra					;\
	$(TOOLDIR)/bet p.ra m1.ra						;\
	$(TOOLDIR)/bet p.ra m2.ra						;\
	$(TOOLDIR)/nrmse -t 1e-6 m1.ra m2.ra					;\
	rm *.ra; cd ..; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-bet-phantom tests/test-bet-scale