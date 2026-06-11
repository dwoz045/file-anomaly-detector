rule HighEntropyExecutable
{
    meta:
        description = "Detects executable files with suspicious high-entropy content indicative of packing or encryption"
        severity = "HIGH"

    strings:
        $mz = { 4D 5A }

    condition:
        $mz at 0 and filesize > 10KB
}

rule SuspiciousShellScript
{
    meta:
        description = "Detects shell scripts containing common reverse shell patterns"
        severity = "HIGH"

    strings:
        $bash_i    = "/bin/bash -i"              ascii
        $dev_tcp   = "/dev/tcp/"                 ascii
        $nc_e      = "nc -e"                     ascii
        $python_s  = "python -c 'import socket"  ascii
        $perl_s    = "perl -e 'use Socket"       ascii

    condition:
        any of them
}

rule Base64EncodedPayload
{
    meta:
        description = "Detects suspiciously long base64 strings which may indicate obfuscated payloads"
        severity = "MEDIUM"

    strings:
        $b64 = /[A-Za-z0-9+\/]{200,}={0,2}/

    condition:
        $b64
}

rule RansomwareNote
{
    meta:
        description = "Detects common ransomware note patterns"
        severity = "CRITICAL"

    strings:
        $r1 = "YOUR FILES HAVE BEEN ENCRYPTED"  ascii nocase
        $r2 = "your files are encrypted"         ascii nocase
        $r3 = "to decrypt your files"            ascii nocase
        $r4 = "bitcoin"                           ascii nocase
        $r5 = "send payment"                      ascii nocase
        $r6 = "DECRYPT_INSTRUCTIONS"             ascii nocase

    condition:
        2 of them
}

rule SuspiciousCronEntry
{
    meta:
        description = "Detects cron entries that download and execute remote scripts"
        severity = "HIGH"

    strings:
        $curl_sh   = "curl"    ascii
        $wget_sh   = "wget"    ascii
        $pipe_sh   = "| sh"    ascii
        $pipe_bash = "| bash"  ascii

    condition:
        ($curl_sh or $wget_sh) and ($pipe_sh or $pipe_bash)
}
