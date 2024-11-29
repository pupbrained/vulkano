{
  description = "C/C++ environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    rust-overlay.url = "github:oxalica/rust-overlay";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = {...} @ inputs:
    with inputs;
      utils.lib.eachDefaultSystem (
        system: let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [(import rust-overlay)];
            config = {
              allowUnfree = true;
              allowUnsupportedSystem = true;
            };
          };

          stdenv =
            if pkgs.hostPlatform.isLinux
            then pkgs.stdenvAdapters.useMoldLinker pkgs.llvmPackages_18.libcxxStdenv
            else pkgs.llvmPackages_18.libcxxStdenv;

          rust = pkgs.rust-bin.nightly.latest.default.override {
            extensions = ["rust-analyzer" "rust-src"];
          };

          deps = with pkgs; [
            libxkbcommon
            vulkan-extension-layer
            vulkan-memory-allocator
            vulkan-utility-libraries
            vulkan-loader
            vulkan-tools
            wayland
            valgrind
          ];
        in
          with pkgs; {
            formatter = treefmt-nix.lib.mkWrapper pkgs {
              projectRootFile = "flake.nix";
              programs = {
                alejandra.enable = true;

                rustfmt.enable = true;
                rustfmt.package = rust;
              };
            };

            devShell = mkShell.override {inherit stdenv;} {
              buildInputs =
                [
                  alejandra
                  linuxKernel.packages.linux_xanmod.perf.out
                  rust
                ]
                ++ deps;

              LD_LIBRARY_PATH = "${lib.makeLibraryPath deps}";
              SHADERC_LIB_DIR = "${pkgs.shaderc.lib}/lib";

              ${
                if stdenv.isDarwin
                then "VK_ICD_FILENAMES"
                else null
              } = "${darwin.moltenvk}/share/vulkan/icd.d/MoltenVK_icd.json";

              name = "Vulkan";
            };
          }
      );
}
