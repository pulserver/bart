
tests/test-extractdc: phantom traj resize extractdc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -o2 -r -x128 -y3 -c trj1.ra						;\
	$(TOOLDIR)/traj -o2 -r -x128 -y3 -q0.2:0.3:0 trj2.ra					;\
	$(TOOLDIR)/phantom -k -ttrj1.ra	ksp1.ra							;\
	$(TOOLDIR)/phantom -k -ttrj2.ra	ksp2.ra							;\
	$(TOOLDIR)/resize -c 1 1 ksp1.ra dc1.ra							;\
	$(TOOLDIR)/extractdc trj2.ra ksp2.ra dc2.ra						;\
	$(TOOLDIR)/nrmse -t 05.e-6 dc1.ra dc2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-extractdc
