nextflow.enable.dsl=2

/*
 * Preliminary Step 0 -> Step 3 VAEQL workflow.
 *
 * The AWS Batch deployment profile must provide the Batch queue, workDir,
 * region, container/job-definition settings, and the read-only S3 Mountpoint
 * volume. This workflow keeps the mounted PDS reference as a `val` input so
 * Nextflow does not stage or copy that reference through a `path` channel.
 *
 * Artifact directories are ordinary `path` outputs and therefore travel
 * between processes through Dataflow Channels. The Step 1 command must create:
 *   preprocessed_bundle/preprocessed.csv
 *   preprocessed_bundle/feature_dict.json
 * The Step 3 command is still an explicit adapter because a DRL entry point is
 * not present in the repository yet. It must create the declared final output
 * directories and evaluation statistics file.
 *
 * Example AWS launch, after supplying a deployment config and commands:
 *   nextflow run VAEQL_plus/conf/nextflow_conf.nf \
 *     -c VAEQL_plus/conf/nextflow_aws.config \
 *     --reference_pds_mount /mnt/pds/trial.csv \
 *     --phase1_manifest_uri s3://bucket/metadata/trial.json \
 *     --step1_command 'python -m <approved_step1_entrypoint>' \
 *     --step3_command 'python -m <approved_step3_entrypoint>'
 */

params.run_id                 = 'local-run'
params.reference_pds_mount    = null
params.metadata_uris          = []
params.phase1_manifest_uri    = null
params.model_name             = 'mistralai/Ministral-8B-Instruct-2410'
params.python                 = 'python'
params.cv_config              = "${projectDir}/CV_beta_C_tuning_params.json"
params.publish_dir            = "${projectDir}/runs/nextflow_outputs"

// These defaults make local graph validation possible; production uses awsbatch.
params.executor               = 'awsbatch'
params.batch_queue            = 'CHANGE_ME'
params.step1_command          = null
params.step3_command          = null

params.step0_cpus             = 4
params.step0_memory           = '32 GB'
params.step0_time             = '12h'
params.step1_cpus             = 8
params.step1_memory           = '32 GB'
params.step1_time             = '12h'
params.step2_cpus             = 8
params.step2_memory           = '64 GB'
params.step2_time             = '48h'
params.step3_cpus             = 8
params.step3_memory           = '64 GB'
params.step3_time             = '48h'


def shell_quote(value) {
    def text = value.toString().replace("'", "'\"'\"'")
    return "'${text}'"
}


process STEP0_SLM_METADATA_PROFILING {
    tag "${run_id}:step0"
    label 'vaeql_gpu'
    executor params.executor
    queue params.batch_queue
    cpus params.step0_cpus
    memory params.step0_memory
    time params.step0_time
    accelerator 1
    errorStrategy 'terminate'
    maxRetries 0

    input:
    tuple val(run_id), val(reference_pds), val(metadata_uris), val(phase1_manifest_uri), val(model_name)

    output:
    tuple val(run_id), val(reference_pds), path('phase1'), emit: phase1_artifacts

    stub:
    """
    mkdir -p phase1
    printf '{}\\n' > phase1/manifest.json
    printf 's3://stub/phase1/manifest.json\\n' > phase1/manifest_uri.txt
    """

    script:
    if (!reference_pds.toString().startsWith('/')) {
        throw new IllegalArgumentException(
            'reference_pds_mount must be an absolute AWS S3 Mountpoint path'
        )
    }
    def metadata_args = (metadata_uris ?: []).collect { "--metadata-uri ${shell_quote(it)}" }.join(' ')
    def metadata_suffix = metadata_args ? " \\\n        ${metadata_args}" : ''
    """
    set -euo pipefail
    test -f ${shell_quote(reference_pds)}
    mkdir -p phase1

    ${params.python} -m VAEQL_plus.step0_SLM_metadata_profiling.feature_type_profiling \
        --input-path ${shell_quote(reference_pds)} \
        --output-uri ${shell_quote(phase1_manifest_uri)} \
        --manifest-path phase1/manifest.json \
        --model-name ${shell_quote(model_name)}${metadata_suffix}

    printf '%s\\n' ${shell_quote(phase1_manifest_uri)} > phase1/manifest_uri.txt
    test -s phase1/manifest.json
    """
}


process STEP1_PREPROCESSING {
    tag "${run_id}:step1"
    label 'vaeql_gpu'
    executor params.executor
    queue params.batch_queue
    cpus params.step1_cpus
    memory params.step1_memory
    time params.step1_time
    accelerator 1
    errorStrategy 'terminate'
    maxRetries 0

    input:
    tuple val(run_id), val(reference_pds), path('phase1')

    output:
    tuple val(run_id), val(reference_pds), path('preprocessed_bundle'), emit: preprocessed_artifacts

    stub:
    """
    mkdir -p preprocessed_bundle
    printf 'feature\\n' > preprocessed_bundle/preprocessed.csv
    printf '{}\\n' > preprocessed_bundle/feature_dict.json
    """

    script:
    if (!params.step1_command) {
        throw new IllegalArgumentException(
            'params.step1_command is required because Step 1 has no stable CLI entry point yet'
        )
    }
    """
    set -euo pipefail
    test -f ${shell_quote(reference_pds)}
    test -s phase1/manifest.json
    mkdir -p preprocessed_bundle

    ${params.step1_command} \
        --reference-pds ${shell_quote(reference_pds)} \
        --phase1-dir phase1 \
        --output-dir preprocessed_bundle

    test -s preprocessed_bundle/preprocessed.csv
    test -s preprocessed_bundle/feature_dict.json
    """
}


process STEP2_BETA_C_TUNING {
    tag "${run_id}:step2"
    label 'vaeql_gpu'
    executor params.executor
    queue params.batch_queue
    cpus params.step2_cpus
    memory params.step2_memory
    time params.step2_time
    accelerator 1
    errorStrategy 'terminate'
    maxRetries 0

    input:
    tuple val(run_id), val(reference_pds), path('preprocessed_bundle')
    path cv_config

    output:
    tuple val(run_id), val(reference_pds), path('model_artifacts'), path('gradients'), path('q_tables'), path('beta_analysis.csv'), path('best_beta_C_summary.json'), emit: tuning_artifacts

    stub:
    """
    mkdir -p model_artifacts gradients q_tables
    printf 'beta,C,score\\n' > beta_analysis.csv
    printf '{}\\n' > best_beta_C_summary.json
    """

    script:
    // The current fine_tuner consumes the reference PDS and the reviewed
    // feature dictionary. The preprocessed bundle remains a channel artifact;
    // a future tuning adapter can consume preprocessed.csv directly.
    """
    set -euo pipefail
    test -f ${shell_quote(reference_pds)}
    test -s preprocessed_bundle/preprocessed.csv
    test -s preprocessed_bundle/feature_dict.json
    mkdir -p model_artifacts gradients q_tables

    ${params.python} -m VAEQL_plus.step2_beta_C_tuning.fine_tuner \
        --input_csv ${shell_quote(reference_pds)} \
        --feature_dict_json preprocessed_bundle/feature_dict.json \
        --cv_config ${cv_config} \
        --results_path beta_analysis.csv \
        --model_outdir model_artifacts \
        --summary_json best_beta_C_summary.json
    """
}


process STEP3_DRL_TRAINING {
    tag "${run_id}:step3"
    label 'vaeql_gpu'
    executor params.executor
    queue params.batch_queue
    cpus params.step3_cpus
    memory params.step3_memory
    time params.step3_time
    accelerator 1
    errorStrategy 'terminate'
    maxRetries 0
    publishDir params.publish_dir, mode: 'copy', overwrite: true

    input:
    tuple val(run_id), val(reference_pds), path('model_artifacts'), path('gradients'), path('q_tables'), path('beta_analysis.csv'), path('best_beta_C_summary.json')

    output:
    tuple val(run_id), path('final_model'), path('run_config'), path('evaluation'), emit: final_artifacts

    stub:
    """
    mkdir -p final_model run_config evaluation
    touch final_model/model.pt
    printf '{}\\n' > run_config/config.json
    printf '{}\\n' > evaluation/evaluation_statistics.json
    """

    script:
    if (!params.step3_command) {
        throw new IllegalArgumentException(
            'params.step3_command is required because the Step 3 DRL entry point is not present yet'
        )
    }
    """
    set -euo pipefail
    test -f ${shell_quote(reference_pds)}
    test -s beta_analysis.csv
    test -s best_beta_C_summary.json
    mkdir -p final_model run_config evaluation

    ${params.step3_command} \
        --reference-pds ${shell_quote(reference_pds)} \
        --model-dir model_artifacts \
        --gradients-dir gradients \
        --q-tables-dir q_tables \
        --beta-summary best_beta_C_summary.json \
        --output-dir .

    test -d final_model
    test -d run_config
    test -s evaluation/evaluation_statistics.json
    """
}


workflow {
    if (!params.reference_pds_mount) {
        error 'reference_pds_mount is required and must point to an AWS S3 Mountpoint file'
    }
    if (!params.phase1_manifest_uri) {
        error 'phase1_manifest_uri is required for the SSE-C-protected Step 0 manifest output'
    }
    if (!params.step1_command) {
        error 'step1_command is required for the preliminary Step 1 adapter'
    }
    if (!params.step3_command) {
        error 'step3_command is required for the preliminary Step 3 adapter'
    }

    def metadata_uris = params.metadata_uris instanceof List
        ? params.metadata_uris
        : (params.metadata_uris ? params.metadata_uris.toString().split(',')*.trim() : [])

    phase1_input = channel.value(tuple(
        params.run_id.toString(),
        params.reference_pds_mount.toString(),
        metadata_uris,
        params.phase1_manifest_uri.toString(),
        params.model_name.toString(),
    ))
    phase1 = STEP0_SLM_METADATA_PROFILING(phase1_input)

    preprocessed = STEP1_PREPROCESSING(phase1)

    cv_config = channel.value(file(params.cv_config, checkIfExists: true))
    tuning = STEP2_BETA_C_TUNING(preprocessed, cv_config)

    STEP3_DRL_TRAINING(tuning)
}
