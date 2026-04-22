tests/test-cc-headers: tests/cc-headers.cc
	$(CXX) -c -I src tests/cc-headers.cc -o /dev/null
	touch $@

TESTS_CXX += tests/test-cc-headers
