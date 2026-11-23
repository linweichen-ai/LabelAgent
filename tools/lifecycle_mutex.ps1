function Enter-LabelAgentLifecycleMutex {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root
    )

    $rootBytes = [Text.Encoding]::UTF8.GetBytes(([IO.Path]::GetFullPath($Root).ToLowerInvariant()))
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $rootHashBytes = $sha256.ComputeHash($rootBytes)
    } finally {
        $sha256.Dispose()
    }
    $rootHash = ([BitConverter]::ToString($rootHashBytes)).Replace("-", "").Substring(0, 24)
    $mutex = [Threading.Mutex]::new($false, "Local\LabelAgentLifecycle-$rootHash")
    $acquired = $false
    try {
        try {
            $acquired = $mutex.WaitOne(0)
        } catch [Threading.AbandonedMutexException] {
            $acquired = $true
        }
        if (-not $acquired) {
            throw "Another LabelAgent lifecycle action is already running for this checkout."
        }
        return $mutex
    } catch {
        if ($acquired) {
            $mutex.ReleaseMutex()
        }
        $mutex.Dispose()
        throw
    }
}

function Exit-LabelAgentLifecycleMutex {
    param(
        [Parameter(Mandatory = $true)]
        [Threading.Mutex]$Mutex
    )

    try {
        $Mutex.ReleaseMutex()
    } finally {
        $Mutex.Dispose()
    }
}
