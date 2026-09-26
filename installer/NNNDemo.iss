#define AppName "NNNDemo"
#define Publisher "Marco Vasko"
#define ExecutableName "NNNDemo.exe"

[Setup]
AppId={{23602532adb74e89869dc27f9c54ac23}}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}

DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}

OutputDir=output
OutputBaseFilename=NNNDemo-v{#AppVersion}

ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

PrivilegesRequired=admin

Compression=lzma
SolidCompression=yes

WizardStyle=modern

UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#ExecutableName}

[Files]
Source: "{#ExecutablePath}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ManagedPath}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#NativePath}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ModelsDir}\*"; DestDir: "{app}\Models"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#ExecutableName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#ExecutableName}"; WorkingDir: "{app}"

[Run]
Filename: "{app}\{#ExecutableName}"; WorkingDir: "{app}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
