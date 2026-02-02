{
  description = "Offline Speech-to-Text with Whisper";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python313;

        pythonEnv = python.withPackages (ps: with ps; [
          faster-whisper
          tqdm
          loguru
          psutil
          textual
          sounddevice
        ]);

        runtimeDeps = with pkgs; [
          sox
          ffmpeg
          portaudio
          libsndfile
        ];

        offlinestt = pkgs.stdenv.mkDerivation {
          pname = "offlinestt";
          version = "0.1.0";

          src = ./.;

          nativeBuildInputs = [ pkgs.makeWrapper ];

          buildInputs = [ pythonEnv ] ++ runtimeDeps;

          installPhase = ''
            mkdir -p $out/bin $out/share/applications $out/share/icons/hicolor/scalable/apps $out/lib/offlinestt

            # Copy Python files
            cp tui.py transcribe.py $out/lib/offlinestt/

            # Create wrapper script
            makeWrapper ${pythonEnv}/bin/python $out/bin/offlinestt \
              --add-flags "$out/lib/offlinestt/tui.py" \
              --prefix PATH : ${pkgs.lib.makeBinPath runtimeDeps} \
              --prefix LD_LIBRARY_PATH : ${pkgs.lib.makeLibraryPath [
                pkgs.portaudio
                pkgs.libsndfile
                pkgs.stdenv.cc.cc.lib
              ]}

            # Install icon
            cp icon.svg $out/share/icons/hicolor/scalable/apps/offlinestt.svg

            # Create desktop entry
            cat > $out/share/applications/offlinestt.desktop << EOF
            [Desktop Entry]
            Name=Offline STT
            Comment=Record and transcribe audio using Whisper
            Exec=$out/bin/offlinestt
            Icon=offlinestt
            Terminal=true
            Type=Application
            Categories=AudioVideo;Audio;Utility;
            Keywords=speech;transcription;whisper;audio;recording;
            EOF
          '';

          meta = with pkgs.lib; {
            description = "Offline Speech-to-Text using Whisper";
            homepage = "https://github.com/antonym/offlinestt";
            license = licenses.mit;
            platforms = platforms.linux;
            mainProgram = "offlinestt";
          };
        };

      in
      {
        packages = {
          default = offlinestt;
          offlinestt = offlinestt;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pkgs.uv
          ] ++ runtimeDeps ++ [
            pkgs.stdenv.cc.cc.lib
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath ([
              pkgs.portaudio
              pkgs.libsndfile
              pkgs.stdenv.cc.cc.lib
            ])}:$LD_LIBRARY_PATH"
            
            echo "Offline STT dev environment loaded"
            echo "Python: $(python --version)"
            echo "Run 'uv sync' to install Python dependencies"
          '';
        };
      });
}
