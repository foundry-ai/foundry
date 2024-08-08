{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let lib = nixpkgs.lib;
    setuptools = build-system.setuptools;
    stdenv = nixpkgs.stdenv;
    CoreFoundation = nixpkgs.darwin.apple_sdk.frameworks.CoreFoundation;
    fetchPypi = python.pkgs.fetchPypi;
    IOKit = nixpkgs.darwin.IOKit;
    pytestCheckHook = python.pkgs.pytestCheckHook;
    pythonOlder = python.pkgs.pythonOlder;
in
buildPythonPackage rec {
  pname = "psutil";
  version = "6.0.0";
  format = "setuptools";

  inherit stdenv;

  disabled = pythonOlder "3.7";

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-j6rk8xC22Wn6JsoFRTOLIfc8axXbfEqNk0pUgvqoGPI=";
  };

  build-system = [setuptools];

  postPatch = ''
    # stick to the old SDK name for now
    # https://developer.apple.com/documentation/iokit/kiomasterportdefault/
    # https://developer.apple.com/documentation/iokit/kiomainportdefault/
    substituteInPlace psutil/arch/osx/cpu.c \
      --replace-fail kIOMainPortDefault kIOMasterPortDefault
  '';

  buildInputs =
    # workaround for https://github.com/NixOS/nixpkgs/issues/146760
    lib.optionals (stdenv.isDarwin && stdenv.isx86_64) [ CoreFoundation ]
    ++ lib.optionals stdenv.isDarwin [ IOKit ];

  nativeCheckInputs = [ pytestCheckHook ];
  doCheck = false;
  pytestFlagsArray = [
    "$out/${python.sitePackages}/psutil/tests/test_system.py"
  ];
  disabledTests = [
    # Some of the tests have build-system hardware-based impurities (like
    # reading temperature sensor values).  Disable them to avoid the failures
    # that sometimes result.
    "cpu_freq"
    "cpu_times"
    "disk_io_counters"
    "sensors_battery"
    "sensors_temperatures"
    "user"
    "test_disk_partitions" # problematic on Hydra's Linux builders, apparently
  ];

  pythonImportsCheck = [ "psutil" ];

  meta = with lib; {
    description = "Process and system utilization information interface";
    homepage = "https://github.com/giampaolo/psutil";
    changelog = "https://github.com/giampaolo/psutil/blob/release-${version}/HISTORY.rst";
    license = licenses.bsd3;
    maintainers = [ ];
  };
}