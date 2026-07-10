
tests/test-seq-raga: seq traj extract nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -r -D trj_ref.ra 	;\
	$(TOOLDIR)/seq -r 377 --raga samples.ra grad.ra mom.ra 	;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra		;\
	$(TOOLDIR)/nrmse -t 3E-7 trj_ref.ra trj_seq.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-raga


tests/test-seq-raga2: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -r -D trj_ref.ra 	;\
	$(TOOLDIR)/seq -r 377 --dwell 4.2E-6 --raga samples.ra grad.ra mom.ra 	;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra		;\
	$(TOOLDIR)/nrmse -t 3E-7 trj_ref.ra trj_seq.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-raga2


tests/test-seq-raga-chrono: seq traj extract nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -r -A -s 1 --double-base trj_ref.ra ;\
	$(TOOLDIR)/seq -r 377 --raga --chrono samples.ra		 	;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra				;\
	$(TOOLDIR)/nrmse -t 3E-7 trj_ref.ra trj_seq.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-raga-chrono


tests/test-seq-raga-sms: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -m3 -r -A -s 1 --double-base trj_ref.ra 	;\
	$(TOOLDIR)/seq -r 377 --raga --chrono --mb_factor=3 -m3 samples.ra		;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra					;\
	$(TOOLDIR)/nrmse -t 3E-7 trj_ref.ra trj_seq.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-raga-sms


tests/test-seq-raga-sms-al: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -m3 -r -A -s 1 -l --double-base trj_ref.ra 		;\
	$(TOOLDIR)/seq -r 377 --raga --raga_flags 8192 --chrono --mb_factor=3 -m3 samples.ra	;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra						;\
	$(TOOLDIR)/nrmse -t 3E-7 trj_ref.ra trj_seq.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-raga-sms-al


tests/test-seq-raga-sms-al-frame: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -m3 -r -A -s 1 -l -t3 --double-base trj_ref.ra	;\
	$(TOOLDIR)/seq -r 377 -t1131 --raga --raga_flags 8192 --chrono --mb_factor=3 -m3 samples.ra		;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra						;\
	$(TOOLDIR)/nrmse -t 3E-7 trj_ref.ra trj_seq.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-raga-sms-al-frame


tests/test-seq-offcenter: seq traj extract scale phantom fovshift fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -r -A -s 1 --double-base trj_ref.ra		;\
	$(TOOLDIR)/seq -s 0.0256:0.0128:0 -r 377 --raga --chrono --no-spoiling samples.ra	;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra					;\
	$(TOOLDIR)/extract 0 4 5 samples.ra adc_phase.ra				;\
	$(TOOLDIR)/scale 0.5 trj_ref.ra trj_scale.ra					;\
	$(TOOLDIR)/phantom -k -t trj_scale.ra ksp.ra					;\
	$(TOOLDIR)/fovshift -s 0.1:0.05:0 -t trj_ref.ra ksp.ra ksp_ref.ra		;\
	$(TOOLDIR)/fmac adc_phase.ra ksp.ra ksp_seq.ra					;\
	$(TOOLDIR)/nrmse -t 1E-6 ksp_ref.ra ksp_seq.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-offcenter


tests/test-seq-cartesian: seq traj extract nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 256 -c trj_ref.ra 			;\
	$(TOOLDIR)/seq -r 256 --cartesian --TE 2.2E-3 --TR 4E-3 samples.ra 	;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra				;\
	$(TOOLDIR)/nrmse -t 1e-6 trj_ref.ra trj_seq.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-cartesian


tests/test-seq-offcenter-cart-ro: seq traj extract phantom fovshift fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/seq -S 0.05:0:0 -r 256 --cartesian --TE 2.2E-3 --TR 4E-3 --no-spoiling samples.ra 	;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra			;\
	$(TOOLDIR)/extract 0 4 5 samples.ra adc_phase.ra		;\
	$(TOOLDIR)/phantom -t trj_seq.ra -k ksp.ra			;\
	$(TOOLDIR)/fovshift -s 0.:0.025:0 ksp.ra ksp_ref.ra		;\
	$(TOOLDIR)/fmac adc_phase.ra ksp.ra ksp_seq.ra			;\
	$(TOOLDIR)/nrmse -S -t 1.5e-6 ksp_ref.ra ksp_seq.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-offcenter-cart-ro

tests/test-seq-offcenter-cart-pe: seq traj extract phantom fovshift fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/seq -S 0:0.1:0 -r 256 --cartesian --TE 2.2E-3 --TR 4E-3 --no-spoiling samples.ra 	;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra			;\
	$(TOOLDIR)/extract 0 4 5 samples.ra adc_phase.ra		;\
	$(TOOLDIR)/phantom -t trj_seq.ra -k ksp.ra			;\
	$(TOOLDIR)/fovshift -t trj_seq.ra -s 0:0.1:0 ksp.ra ksp_ref.ra	;\
	$(TOOLDIR)/fmac adc_phase.ra ksp.ra ksp_seq.ra			;\
	$(TOOLDIR)/nrmse -t 2e-6 ksp_ref.ra ksp_seq.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-offcenter-cart-pe


tests/test-seq-relative-fovshift: seq traj extract scale phantom fovshift fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/seq -s 0.0256:0.0128:0	-r 377 --no-spoiling samples_abs.ra	;\
	$(TOOLDIR)/seq -S 0.1:0.05:0 		-r 377 --no-spoiling samples_rel.ra	;\
	$(TOOLDIR)/nrmse -t 0 samples_abs.ra samples_rel.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-relative-fovshift


tests/test-seq-raga-ordering: seq traj extract raga bin nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/seq -r 377 --raga samples.ra grad.ra mom.ra 			;\
	$(TOOLDIR)/seq -r 377 --raga --chrono samples_c.ra grad_c.ra mom_c.ra	;\
	$(TOOLDIR)/raga 377 ind.ra						;\
	$(TOOLDIR)/bin -o ind.ra grad_c.ra grad_c_sort.ra			;\
	$(TOOLDIR)/nrmse -t 0 grad.ra grad_c_sort.ra				;\
	$(TOOLDIR)/bin -o ind.ra mom_c.ra mom_c_sort.ra				;\
	$(TOOLDIR)/nrmse -t 0 mom.ra mom_c_sort.ra				;\
	$(TOOLDIR)/bin -o ind.ra samples_c.ra samples_c_sort.ra			;\
	$(TOOLDIR)/nrmse -t 0 samples.ra samples_c_sort.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-raga-ordering


tests/test-seq-raga-ind: seq raga nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/raga -s1 377 ind_ref.ra		;\
	$(TOOLDIR)/seq -r 377 --raga -R ind_seq.ra  	;\
	$(TOOLDIR)/nrmse -t 0. ind_ref.ra ind_seq.ra	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-raga-ind


tests/test-seq-raga-ind-multislice: seq raga nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/raga -s1 -m3 377 ind_ref.ra		;\
	$(TOOLDIR)/seq -r 377 -m3 --raga -R ind_seq.ra  ;\
	$(TOOLDIR)/nrmse -t 0. ind_ref.ra ind_seq.ra	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-raga-ind-multislice


tests/test-seq-traj-meco: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -r -D -G -s1 -E -x 220 -o 2. -y5 -e7 -t10 trj_ref.ra	;\
	$(TOOLDIR)/seq --FOV 0.220  --BR 220  --dwell 5.4E-6 --rf_duration 400E-6 --BWTP 1.00 --slice_thickness 5E-3 --TR 13.8E-3 --TE 1.8E-3 --TE_delta 1.8E-3 --mems -r 5 -t 10 -e 7 samples.ra ;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 trj_ref.ra trj_seq.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-traj-meco


tests/test-seq-traj-meco2: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -r -D -G -s1 -E -x 256 -o 2. -y1 -e7 -t1 trj_ref.ra	;\
	$(TOOLDIR)/seq --FOV 0.210  --BR 256  --dwell 5.6E-6 --rf_duration 900E-6 --BWTP 3.8 --slice_thickness 5E-3 --TR 20.3E-3 --TE 2.31E-3 --TE_delta 2.61E-3 --mems -r 1 -t 1 -e 7 samples.ra ;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 trj_ref.ra trj_seq.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-traj-meco2



tests/test-seq-meco-chrono: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -x 220 -o 2. -y 411 -s 31 -e 7 -r -D -A --double-base trj_ref.ra 	;\
	$(TOOLDIR)/seq --FOV 0.220  --BR 220  --dwell 5.4E-6 --rf_duration 400E-6 --BWTP 1.00 --slice_thickness 5E-3 --TR 13.8E-3 --TE 1.8E-3 --TE_delta 1.8E-3 --raga --chrono --tiny 31 -r 411 -e 7 samples.ra ;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra						;\
	$(TOOLDIR)/nrmse -t 5.E-7 trj_ref.ra trj_seq.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-meco-chrono


tests/test-seq-meco: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -x 220 -o 2. -y 411 -e 7 -r -D trj_ref.ra 				;\
	$(TOOLDIR)/seq --FOV 0.220  --BR 220  --dwell 5.4E-6 --rf_duration 400E-6 --BWTP 1.00 --slice_thickness 5E-3 --TR 13.8E-3 --TE 1.8E-3 --TE_delta 1.8E-3 --raga --tiny 31 -r 411 -e 7 samples.ra ;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra						;\
	$(TOOLDIR)/nrmse -t 5.E-7 trj_ref.ra trj_seq.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-meco


tests/test-seq-raga-ind-meco: seq raga nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/raga -s20 -e7 433 ind_ref.ra					;\
	$(TOOLDIR)/seq -r 433 -e7 --tiny 20  --TR 20e-3 --raga -R ind_seq.ra  	;\
	$(TOOLDIR)/nrmse -t 0. ind_ref.ra ind_seq.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-raga-ind-meco


tests/test-seq-asym: traj seq extract nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x 192 -d 256 -o 2. -y 377 -r -D trj_ref.ra 		;\
	$(TOOLDIR)/seq -r 377 --asym_echo 0.25 --raga samples.ra grad.ra mom.ra ;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra				;\
	$(TOOLDIR)/nrmse -t 2E-7 trj_ref.ra trj_seq.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-asym


tests/test-seq-spoiled-raga: seq traj extract nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -r -D trj_ref.ra 	;\
	$(TOOLDIR)/seq -r 377 --spoiled --TR 5.25E-3 --raga samples.ra grad.ra mom.ra 	;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra		;\
	$(TOOLDIR)/nrmse -t 3E-7 trj_ref.ra trj_seq.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-seq-raga
