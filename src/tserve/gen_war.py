from workflow_archiver.workflow_packaging import package_workflow
from workflow_archiver.workflow_packaging_utils import WorkflowExportUtils
from workflow_archiver.workflow_archiver_error import WorkflowArchiverError
from argparse import Namespace


workflow_spec_file = "src/tserve/workflow.yaml"
workflow_name = "wf"
handler = "src/tserve/wf_handler.py"
export_file_path = "mar_files/"
extra_files = None


WorkflowExportUtils.validate_inputs(workflow_name, export_file_path)
args = Namespace(
    **dict(
        spec_file=workflow_spec_file,
        workflow_name=workflow_name,
        handler=handler,
        export_path=export_file_path,
        extra_files=extra_files,
        force=False
    )
)
manifest = WorkflowExportUtils.generate_manifest_json(args)
package_workflow(args, manifest=manifest)
