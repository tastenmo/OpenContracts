import {
  ModalContent,
  Button,
  Modal,
  Icon,
  Dimmer,
  Loader,
} from "semantic-ui-react";
import {
  ColumnType,
  DatacellType,
  DocumentType,
  ExtractType,
} from "../../../types/graphql-api";
import {
  useMutation,
  useQuery,
  useReactiveVar,
  NetworkStatus,
} from "@apollo/client";
import {
  RequestGetExtractOutput,
  REQUEST_GET_EXTRACT,
  RequestGetExtractInput,
} from "../../../graphql/queries";
import { useEffect, useState, useCallback, useRef } from "react";
import {
  REQUEST_ADD_DOC_TO_EXTRACT,
  REQUEST_CREATE_COLUMN,
  REQUEST_DELETE_COLUMN,
  REQUEST_REMOVE_DOC_FROM_EXTRACT,
  REQUEST_START_EXTRACT,
  RequestAddDocToExtractInputType,
  RequestAddDocToExtractOutputType,
  RequestCreateColumnInputType,
  RequestCreateColumnOutputType,
  RequestDeleteColumnInputType,
  RequestDeleteColumnOutputType,
  RequestRemoveDocFromExtractInputType,
  RequestRemoveDocFromExtractOutputType,
  RequestStartExtractInputType,
  RequestStartExtractOutputType,
  REQUEST_CREATE_FIELDSET,
  RequestCreateFieldsetInputType,
  RequestCreateFieldsetOutputType,
  REQUEST_UPDATE_EXTRACT,
  RequestUpdateExtractInputType,
  RequestUpdateExtractOutputType,
} from "../../../graphql/mutations";
import { toast } from "react-toastify";
import {
  addingColumnToExtract,
  editingColumnForExtract,
} from "../../../graphql/cache";
import {
  ExtractDataGrid,
  ExtractDataGridHandle,
} from "../../extracts/datagrid/DataGrid";
import { CSSProperties } from "react";

interface EditExtractModalProps {
  ext: ExtractType | null;
  open: boolean;
  toggleModal: () => void;
}

// Add new styled components at the top
const styles = {
  modalWrapper: {
    height: "90vh",
    display: "flex !important",
    flexDirection: "column",
    background: "#ffffff",
    overflow: "hidden",
    margin: "5vh auto !important",
    maxHeight: "90vh !important",
  } as React.CSSProperties,
  modalHeader: {
    background: "white",
    padding: "1.5rem 2rem",
    borderBottom: "1px solid #e2e8f0",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.05)",
    position: "sticky",
    top: 0,
    zIndex: 10,
    flex: "0 0 auto",
  } as React.CSSProperties,
  modalContent: {
    flex: "1 1 auto",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    minHeight: 0,
    maxHeight: "calc(90vh - 130px) !important",
  } as React.CSSProperties,
  scrollableContent: {
    flex: "1 1 auto",
    display: "flex",
    flexDirection: "column",
    overflow: "auto",
    minHeight: 0,
    padding: "0 2rem",
  } as React.CSSProperties,
  headerTitle: {
    display: "flex",
    alignItems: "center",
    gap: "1rem",
  },
  extractName: {
    fontSize: "1.5rem",
    fontWeight: 600,
    color: "#1e293b",
    margin: 0,
  },
  extractMeta: {
    fontSize: "0.875rem",
    color: "#475569",
  },
  statsContainer: {
    flex: "0 0 auto",
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
    gap: "1rem",
    padding: "1.5rem 0",
    background: "white",
    borderRadius: "0.5rem",
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
    margin: "1rem 0",
  },
  statCard: {
    padding: "1.25rem",
    background: "#ffffff",
    borderRadius: "0.5rem",
    border: "1px solid #cbd5e1",
    transition: "transform 0.2s ease",
    "&:hover": {
      transform: "translateY(-2px)",
    },
  },
  statLabel: {
    fontSize: "0.875rem",
    fontWeight: 500,
    color: "#475569",
    marginBottom: "0.5rem",
  },
  statValue: {
    fontSize: "1.25rem",
    fontWeight: 600,
    color: "#1e293b",
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
  },
  actionButtons: {
    padding: "0.75rem 2rem",
    display: "flex",
    gap: "1rem",
    justifyContent: "flex-end",
    background: "white",
    borderTop: "1px solid #e2e8f0",
  },
  errorMessage: {
    margin: "1rem 2rem",
    padding: "1rem",
    borderRadius: "0.5rem",
    background: "#fee2e2",
    border: "1px solid #fecaca",
    color: "#991b1b",
  },
  dataGridContainer: {
    flex: "1 1 auto",
    minHeight: 0,
    display: "flex",
    flexDirection: "column",
    position: "relative",
    margin: "0 0 1rem",
  } as React.CSSProperties,
  modalActions: {
    flex: "0 0 auto",
    padding: "1rem 2rem !important",
    background: "white",
    borderTop: "1px solid #e2e8f0",
    position: "sticky",
    bottom: 0,
    zIndex: 10,
    boxShadow: "0 -4px 12px rgba(0,0,0,0.05)",
  } as React.CSSProperties,
  startButton: {
    width: "40px",
    height: "40px",
    background: "linear-gradient(135deg, #2563eb, #1d4ed8)",
    transition: "all 0.3s ease",
    border: "none",
    padding: "0",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
    "&:hover": {
      transform: "translateY(-2px)",
      boxShadow: "0 8px 20px -2px rgba(37, 99, 235, 0.35)",
      background: "linear-gradient(135deg, #3b82f6, #2563eb)",
      "& .play-icon": {
        transform: "scale(0)",
        opacity: 0,
      },
      "& .rocket-icon": {
        transform: "scale(1)",
        opacity: 1,
      },
    },
  } as CSSProperties,
  iconBase: {
    position: "absolute",
    fontSize: "16px",
    margin: "0",
    transition: "all 0.3s ease",
  } as CSSProperties,
  playIcon: {
    transform: "scale(1)",
    opacity: 1,
  } as CSSProperties,
  rocketIcon: {
    transform: "scale(0)",
    opacity: 0,
  } as CSSProperties,
  statusWithButton: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    width: "100%",
  },
  downloadButton: {
    background: "transparent",
    border: "1px solid #e2e8f0",
    borderRadius: "8px",
    padding: "8px 16px",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    color: "#64748b",
    fontSize: "0.875rem",
    fontWeight: 500,
    transition: "all 0.2s ease",
    "&:hover:not(:disabled)": {
      background: "#f8fafc",
      borderColor: "#94a3b8",
      transform: "translateY(-1px)",
      color: "#334155",
      boxShadow: "0 2px 4px rgba(148, 163, 184, 0.1)",
    },
    "&:active:not(:disabled)": {
      transform: "translateY(0)",
    },
    "&:disabled": {
      opacity: 0.5,
      cursor: "not-allowed",
    },
  } as CSSProperties,
  downloadIcon: {
    fontSize: "16px",
    transition: "transform 0.2s ease",
  } as CSSProperties,
  controlsContainer: {
    flex: "0 0 auto",
    display: "flex",
    justifyContent: "flex-end",
    gap: "12px",
    padding: "0 0 1rem",
    alignItems: "center",
  } as React.CSSProperties,
};

export const EditExtractModal = ({
  open,
  ext,
  toggleModal,
}: EditExtractModalProps) => {
  const dataGridRef = useRef<ExtractDataGridHandle>(null);

  const [extract, setExtract] = useState<ExtractType | null>(ext);
  const [cells, setCells] = useState<DatacellType[]>([]);
  const [rows, setRows] = useState<DocumentType[]>([]);
  const [columns, setColumns] = useState<ColumnType[]>([]);
  const adding_column_to_extract = useReactiveVar(addingColumnToExtract);
  const editing_column_for_extract = useReactiveVar(editingColumnForExtract);

  useEffect(() => {
    console.log("adding_column_to_extract", adding_column_to_extract);
  }, [adding_column_to_extract]);

  useEffect(() => {
    if (ext) {
      setExtract(ext);
    }
  }, [ext]);

  const [addDocsToExtract, { loading: add_docs_loading }] = useMutation<
    RequestAddDocToExtractOutputType,
    RequestAddDocToExtractInputType
  >(REQUEST_ADD_DOC_TO_EXTRACT, {
    onCompleted: (data) => {
      console.log("Add data to ", data);
      setRows((old_rows) => [
        ...old_rows,
        ...(data.addDocsToExtract.objs as DocumentType[]),
      ]);
      toast.success("SUCCESS! Added docs to extract.");
    },
    onError: (err) => {
      toast.error("ERROR! Could not add docs to extract.");
    },
  });

  const handleAddDocIdsToExtract = (
    extractId: string,
    documentIds: string[]
  ) => {
    addDocsToExtract({
      variables: {
        extractId,
        documentIds,
      },
    });
  };

  const [removeDocsFromExtract, { loading: remove_docs_loading }] = useMutation<
    RequestRemoveDocFromExtractOutputType,
    RequestRemoveDocFromExtractInputType
  >(REQUEST_REMOVE_DOC_FROM_EXTRACT, {
    onCompleted: (data) => {
      toast.success("SUCCESS! Removed docs from extract.");
      console.log("Removed docs and return data", data);
      setRows((old_rows) =>
        old_rows.filter(
          (item) => !data.removeDocsFromExtract.idsRemoved.includes(item.id)
        )
      );
    },
    onError: (err) => {
      toast.error("ERROR! Could not remove docs from extract.");
    },
  });

  const handleRemoveDocIdsFromExtract = (
    extractId: string,
    documentIds: string[]
  ) => {
    removeDocsFromExtract({
      variables: {
        extractId,
        documentIdsToRemove: documentIds,
      },
    });
  };

  const [deleteColumn] = useMutation<
    RequestDeleteColumnOutputType,
    RequestDeleteColumnInputType
  >(REQUEST_DELETE_COLUMN, {
    onCompleted: (data) => {
      toast.success("SUCCESS! Removed column from Extract.");
      setColumns((columns) =>
        columns.filter((item) => item.id !== data.deleteColumn.deletedId)
      );
    },
    onError: (err) => {
      toast.error("ERROR! Could not remove column.");
    },
  });

  const [createFieldset] = useMutation<
    RequestCreateFieldsetOutputType,
    RequestCreateFieldsetInputType
  >(REQUEST_CREATE_FIELDSET);

  const [updateExtract] = useMutation<
    RequestUpdateExtractOutputType,
    RequestUpdateExtractInputType
  >(REQUEST_UPDATE_EXTRACT, {
    onCompleted: () => {
      toast.success("Extract updated with new fieldset.");
      refetch();
    },
    onError: () => {
      toast.error("Failed to update extract with new fieldset.");
    },
  });

  /**
   * Handles the deletion of a column from the extract.
   * If the fieldset is not in use, deletes the column directly.
   * If the fieldset is in use, creates a new fieldset without the column and updates the extract.
   *
   * @param {string} columnId - The ID of the column to delete.
   */
  const handleDeleteColumnIdFromExtract = async (columnId: string) => {
    if (!extract?.fieldset?.id) return;

    if (!extract.fieldset.inUse) {
      // Fieldset is not in use; delete the column directly
      try {
        await deleteColumn({
          variables: {
            id: columnId,
          },
        });
        // Remove the column from local state
        setColumns((prevColumns) =>
          prevColumns.filter((column) => column.id !== columnId)
        );
        // Refetch data to get updated columns
        refetch();
        toast.success("SUCCESS! Removed column from Extract.");
      } catch (error) {
        console.error(error);
        toast.error("Error while deleting column from extract.");
      }
    } else {
      // Fieldset is in use; proceed with existing logic
      try {
        // Step 1: Create a new fieldset
        const { data: fieldsetData } = await createFieldset({
          variables: {
            name: `${extract.fieldset.name} (edited)`,
            description: extract.fieldset.description || "",
          },
        });

        const newFieldsetId = fieldsetData?.createFieldset.obj.id;

        if (!newFieldsetId) throw new Error("Fieldset creation failed.");

        // Step 2: Copy existing columns except the deleted one
        const columnsToCopy = columns.filter((col) => col.id !== columnId);
        await Promise.all(
          columnsToCopy.map((column) =>
            createColumn({
              variables: {
                fieldsetId: newFieldsetId,
                name: column.name,
                query: column.query || "",
                matchText: column.matchText,
                outputType: column.outputType,
                limitToLabel: column.limitToLabel,
                instructions: column.instructions,
                taskName: column.taskName,
                agentic: Boolean(column.agentic),
              },
            })
          )
        );

        // Step 3: Update the extract to use the new fieldset
        console.log("Updating extract to use new fieldset", newFieldsetId);
        await updateExtract({
          variables: {
            id: extract.id,
            fieldsetId: newFieldsetId,
          },
        });

        // Update local state
        setExtract((prevExtract) =>
          prevExtract
            ? { ...prevExtract, fieldset: fieldsetData.createFieldset.obj }
            : prevExtract
        );

        // Refetch data to get updated columns
        refetch();
      } catch (error) {
        console.error(error);
        toast.error("Error while deleting column from extract.");
      }
    }
  };

  const [createColumn, { loading: create_column_loading }] = useMutation<
    RequestCreateColumnOutputType,
    RequestCreateColumnInputType
  >(REQUEST_CREATE_COLUMN, {
    onCompleted: (data) => {
      toast.success("SUCCESS! Created column.");
      setColumns((columns) => [...columns, data.createColumn.obj]);
      addingColumnToExtract(null);
    },
    onError: (err) => {
      toast.error("ERROR! Could not create column.");
      addingColumnToExtract(null);
    },
  });

  // Define the handler for adding a column
  const handleAddColumn = useCallback(() => {
    if (!extract?.fieldset) return;
    addingColumnToExtract(extract);
  }, [extract?.fieldset]);

  const {
    loading,
    error,
    data: extract_data,
    refetch,
    networkStatus,
  } = useQuery<RequestGetExtractOutput, RequestGetExtractInput>(
    REQUEST_GET_EXTRACT,
    {
      variables: {
        id: extract ? extract.id : "",
      },
      nextFetchPolicy: "network-only",
      notifyOnNetworkStatusChange: true,
    }
  );

  useEffect(() => {
    let pollInterval: NodeJS.Timeout;

    if (extract && extract.started && !extract.finished && !extract.error) {
      // Start polling every 5 seconds
      pollInterval = setInterval(() => {
        refetch({ id: extract.id });
      }, 5000);

      // Set up a timeout to stop polling after 10 minutes
      const timeoutId = setTimeout(() => {
        clearInterval(pollInterval);
        toast.info(
          "Job is taking too long... polling paused after 10 minutes."
        );
      }, 600000);

      // Clean up the interval and timeout when the component unmounts or the extract changes
      return () => {
        clearInterval(pollInterval);
        clearTimeout(timeoutId);
      };
    }
  }, [extract, refetch]);

  useEffect(() => {
    if (open && extract) {
      refetch();
    }
  }, [open]);

  useEffect(() => {
    console.log("XOXO - Extract Data", extract_data);
    if (extract_data) {
      const { fullDatacellList, fullDocumentList, fieldset } =
        extract_data.extract;
      console.log("XOXO - Full Datacell List", fullDatacellList);
      console.log("XOXO - Full Document List", fullDocumentList);
      console.log("XOXO - Fieldset", fieldset);
      setCells(fullDatacellList ? fullDatacellList : []);
      setRows(fullDocumentList ? fullDocumentList : []);
      // Add debug logging here
      console.log("Setting columns to:", fieldset?.fullColumnList);
      setColumns(fieldset?.fullColumnList ? fieldset.fullColumnList : []);
      // Update the extract state with the latest data
      setExtract(extract_data.extract);
    }
  }, [extract_data]);

  const [startExtract, { loading: start_extract_loading }] = useMutation<
    RequestStartExtractOutputType,
    RequestStartExtractInputType
  >(REQUEST_START_EXTRACT, {
    onCompleted: (data) => {
      toast.success("SUCCESS! Started extract.");
      setExtract((old_extract) => {
        return { ...old_extract, ...data.startExtract.obj };
      });
    },
    onError: (err) => {
      toast.error("ERROR! Could not start extract.");
    },
  });

  // Add handler for row updates
  const handleRowUpdate = useCallback((updatedRow: DocumentType) => {
    setRows((prevRows) =>
      prevRows.map((row) => (row.id === updatedRow.id ? updatedRow : row))
    );
  }, []);

  // Adjust isLoading to show loading indicator when data is first loading
  const isLoading =
    loading || create_column_loading || add_docs_loading || remove_docs_loading;

  // Determine if the grid should show loading
  const isGridLoading = extract?.started && !extract.finished && !extract.error;

  if (!extract || !extract.id) {
    return null;
  }

  return (
    <>
      <Modal
        id="edit-extract-modal"
        closeIcon
        size="fullscreen"
        open={open}
        onClose={toggleModal}
        style={styles.modalWrapper}
      >
        <div style={styles.modalHeader}>
          <div style={styles.headerTitle}>
            <h2 style={styles.extractName}>{extract.name}</h2>
            <span style={styles.extractMeta}>
              Created by {extract.creator?.email} on{" "}
              {new Date(extract.created).toLocaleDateString()}
            </span>
          </div>
        </div>

        <ModalContent style={styles.modalContent}>
          <div style={styles.scrollableContent}>
            <div style={styles.statsContainer}>
              <div style={styles.statCard}>
                <div style={styles.statLabel}>Status</div>
                <div style={styles.statValue}>
                  {extract.started && !extract.finished && !extract.error ? (
                    <>
                      <Icon name="spinner" loading color="blue" />
                      <span>Processing</span>
                    </>
                  ) : extract.finished ? (
                    <>
                      <Icon name="check circle" color="green" />
                      <span>Completed</span>
                    </>
                  ) : extract.error ? (
                    <>
                      <Icon name="exclamation circle" color="red" />
                      <span>Failed</span>
                    </>
                  ) : (
                    <div style={styles.statusWithButton}>
                      <div>
                        <Icon name="clock outline" color="grey" />
                        <span>Not Started</span>
                      </div>
                      <Button
                        circular
                        icon
                        primary
                        style={styles.startButton}
                        onClick={() =>
                          startExtract({ variables: { extractId: extract.id } })
                        }
                      >
                        <i
                          className="play icon play-icon"
                          style={{ ...styles.iconBase, ...styles.playIcon }}
                        />
                        <i
                          className="rocket icon rocket-icon"
                          style={{ ...styles.iconBase, ...styles.rocketIcon }}
                        />
                      </Button>
                    </div>
                  )}
                </div>
              </div>

              <div style={styles.statCard}>
                <div style={styles.statLabel}>Documents</div>
                <div style={styles.statValue}>
                  <Icon name="file outline" />
                  {rows.length}
                </div>
              </div>

              <div style={styles.statCard}>
                <div style={styles.statLabel}>Columns</div>
                <div style={styles.statValue}>
                  <Icon name="columns" />
                  {columns.length}
                </div>
              </div>

              {extract.corpus && (
                <div style={styles.statCard}>
                  <div style={styles.statLabel}>Corpus</div>
                  <div style={styles.statValue}>
                    <Icon name="database" />
                    {extract.corpus.title}
                  </div>
                </div>
              )}
            </div>

            <div style={styles.controlsContainer}>
              <Button
                basic
                style={styles.downloadButton}
                onClick={() => dataGridRef.current?.exportToCsv()}
                disabled={
                  loading ||
                  isGridLoading ||
                  networkStatus === NetworkStatus.refetch
                }
              >
                <Icon name="download" style={styles.downloadIcon} />
                Export CSV
              </Button>
            </div>

            <div style={{ ...styles.dataGridContainer, position: "relative" }}>
              {loading && (
                <Dimmer
                  active
                  inverted
                  style={{
                    position: "absolute",
                    margin: 0,
                    borderRadius: "12px",
                  }}
                >
                  <Loader>
                    {extract.started && !extract.finished
                      ? "Processing..."
                      : "Loading..."}
                  </Loader>
                </Dimmer>
              )}
              <ExtractDataGrid
                ref={dataGridRef}
                onAddDocIds={handleAddDocIdsToExtract}
                onRemoveDocIds={handleRemoveDocIdsFromExtract}
                onRemoveColumnId={handleDeleteColumnIdFromExtract}
                onUpdateRow={handleRowUpdate}
                onAddColumn={handleAddColumn}
                extract={extract}
                cells={cells}
                rows={rows}
                columns={columns}
                loading={Boolean(isGridLoading)}
              />
            </div>
          </div>
        </ModalContent>

        <div className="actions" style={styles.modalActions}>
          <Button onClick={toggleModal}>Close</Button>
        </div>
      </Modal>
    </>
  );
};
