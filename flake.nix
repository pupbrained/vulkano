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

          deps = with pkgs; [
            vulkan-extension-layer
            vulkan-memory-allocator
            vulkan-utility-libraries
            vulkan-loader
            vulkan-tools
          ];
        in
          with pkgs; {
            formatter = treefmt-nix.lib.mkWrapper pkgs {
              projectRootFile = "flake.nix";
              programs = {
                alejandra.enable = true;

                clang-format = {
                  enable = true;
                  package = pkgs.llvmPackages_18.clang-tools;
                };
              };
            };

            devShell = mkShell.override {inherit stdenv;} {
              buildInputs =
                [
                  alejandra
                  rust-bin.stable.latest.default
                ]
                ++ deps;

              VULKAN_SDK = "${vulkan-headers}";
              VK_LAYER_PATH = "${vulkan-validation-layers}/share/vulkan/explicit_layer.d";
              VK_ICD_FILENAMES =
                if stdenv.isDarwin
                then "${darwin.moltenvk}/share/vulkan/icd.d/MoltenVK_icd.json"
                else let
                  vulkanDir =
                    if stdenv.hostPlatform.isx86_64
                    then "${mesa.drivers}/share/vulkan/icd.d"
                    else "${nixos-asahi.packages.aarch64-linux.mesa-asahi-edge.drivers}/share/vulkan/icd.d";
                  vulkanFiles = builtins.filter (file: builtins.match ".*\\.json$" file != null) (builtins.attrNames (builtins.readDir vulkanDir));
                  vulkanPaths = lib.concatStringsSep ":" (map (file: "${vulkanDir}/${file}") vulkanFiles);
                in
                  if stdenv.hostPlatform.isx86_64
                  then "${linuxPackages_latest.nvidia_x11_beta}/share/vulkan/icd.d/nvidia_icd.x86_64.json:${vulkanPaths}"
                  else vulkanPaths;

              name = "Vulkan";
            };
          }
      );
}
