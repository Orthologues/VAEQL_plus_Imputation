nextflow.enable.dsl=2

/*
 * Cross-validated beta/C halving search for the disentangled Beta-VAE.
 *
 * Usage:
 *   nextflow run CV_tuning_upstream.nf \
 *     --config /abs/path/to/cv_trainer_params.json \
 *     --python python3 \
 *     --publish_dir ./runs/cv_tuning_outputs
 *
 * The config file must define data paths (absolute is safest), beta/C ranges,
 * and output targets (results_path, model_outdir). This workflow simply wraps
 * `DisentangledBetaVAE.py --config <json>` and copies the key artifacts into
 * the publish directory for convenience.
 */

params.config      = "${projectDir}/cv_trainer_params.json"
params.python      = "python"
params.publish_dir = "${projectDir}/runs/cv_tuning_outputs"
params.cpus        = 4
params.memory      = '16 GB'
params.time        = '48h'

process RUN_BETA_C_HALVING {
    tag "config=${config_file.simpleName}"
    cpus params.cpus
    memory params.memory
    time params.time
    errorStrategy 'terminate'
    publishDir params.publish_dir, mode: 'copy', overwrite: true

    input:
    path config_file

    output:
    path "beta_analysis.csv", optional: true
    path "lock.txt", optional: true
    path "trained_models", optional: true

    script:
    """
    set -euo pipefail

    # Keep original work dir so we can stage outputs back for Nextflow.
    WORKDIR="\$PWD"
    CONFIG_PATH="\$(realpath ${config_file})"

    cd ${projectDir}
    ${params.python} DisentangledBetaVAE.py --config "\${CONFIG_PATH}"

    # Stage outputs for Nextflow publishing.
    for f in beta_analysis.csv lock.txt; do
      if [[ -f "\${f}" ]]; then
        cp "\${f}" "\${WORKDIR}/"
      fi
    done

    if [[ -d trained_models ]]; then
      cp -r trained_models "\${WORKDIR}/"
    fi
    """
}

workflow {
    Channel
        .value(file(params.config, checkIfExists: true))
        .set { config_ch }

    RUN_BETA_C_HALVING(config_ch)
}
