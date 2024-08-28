![LakeChime](https://media.licdn.com/dms/image/v2/D4D08AQHiSNC7OBjE_w/croft-frontend-shrinkToFit1024/croft-frontend-shrinkToFit1024/0/1714435200275?e=2147483647&v=beta&t=hdnQsZMaKxWSTQ0a2oG3JR3R3KpmapnEUCmzE-mWe9w) 

Enriched by its vast array of members, skills, organizations, educational institutions, job listings, content, feed, and other interactions, the LinkedIn platform creates billions of data points. This data fuels various applications, from recommendations and rankings to searches, AI functionalities, and other elements enhancing the member experience. All this information is stored in a massive data lake, which allows for quick, scalable, and effective access and processing of extensive datasets. However, this also poses significant infrastructure challenges in gathering, managing, and utilizing this extensive data collection.  

To that end, hundreds of thousands of data pipelines execute daily on our data lake, continually consuming data and writing back insights for further processing by downstream pipelines. Executing pipelines as soon as data is available is crucial for making timely insights; this is easily facilitated by building data triggers that signal the availability of data. These data trigger primitives are typically interrelated to a data lake’s metadata, since most often the metadata is updated once the data is ready to process. In turn, metadata is interrelated with the data lake’s table format. Therefore, table formats used in the data lake play a significant role in deciding data trigger primitives and semantics. In the next section, we discuss table formats, their history, and interrelationship with data triggers in more detail.

## The Evolution and Impact of Table Formats on Data Trigger Mechanisms

Table formats define the structure and organization of data within a data lake, specifying how data is stored, accessed, and managed. Until recently, the [Apache Hive](https://hive.apache.org/) table format has been the format of choice for data lakes. In the Hive table format, data is organized into directories corresponding to partitions. The registration of partition metadata conventionally served as the signal of data arrival and the trigger for executing data pipelines. However, this convention suffered from significant gaps:  

-   **Coarse Granularity:** Data consumption was constrained by the granularity of partition creation. For instance, if partitions were created daily, consumers could only schedule daily jobs to consume new partitions.
-   **Partial Data Consumption:** Partitions are created once, but the data within them can continually be updated. This forces data pipeline owners to choose between registering partitions late (to maximize data visible once the partitions are registered but sacrifice on latency), or registering early (to achieve low latency, but sacrifice on the completeness of data available at registration time).  
    

As data lakes evolve towards modern table formats like [Apache Iceberg](https://iceberg.apache.org/), [Delta](https://delta.io/), and [Apache Hudi](https://hudi.apache.org/), new metadata primitives are becoming more mainstream. One breakthrough that these data formats provide is ACID transactions and semantics. They introduce the concept of “snapshots” to express units of data change. Such units of change represent a much more granular level of metadata, and address the gaps in having to consume partial data, as was the case with the Hive table format. However, some challenges remain as to:

-   How to present the abstraction of data triggers as a concept to the user, decoupling them from the underlying metadata representation differences, whether they are between Hive and the other formats, or among the other formats.
-   How to migrate a data lake that predominantly relies on Hive partition semantics for data triggers to a data lake that is powered by modern table formats, and whose data arrival semantics depend on partitions.
-   How to handle the scale, throughput, and latency of such metadata requests in modern table formats. Hive metadata is powered by a scalable [MySQL](https://www.mysql.com/) backend, while most modern table formats store metadata in distributed file system files with structured data, such as [Avro](https://avro.apache.org/) or [JSON](https://www.json.org/json-en.html).  
    

## Introducing LakeChime: A Unified Data Trigger Solution

In this blog post, we introduce LakeChime, a data trigger service that unifies data trigger semantics not only among modern table formats, but also between modern and traditional table formats such as Hive, bridging the impedance mismatch between traditional partition semantics and modern snapshot semantics. At LinkedIn, we use LakeChime to support data triggers for Hive as well as Iceberg tables (maintained at LinkedIn by [OpenHouse](https://www.openhousedb.org/), the table catalog and control plane). Further, LakeChime is used as one of the main ways to streamline the migration from Hive to Iceberg through its data trigger compatibility layer.  

LakeChime supports both types of data triggers: classical _partition triggers,_ triggering workflows based on the availability of partitions, and modern _snapshot triggers_, triggering workflows based on the availability of new data snapshots. Further, LakeChime is powered by an RDBMS backend, making it ideal to handle large-scale data triggers in very large data lakes. Specifically, LakeChime unlocks the following use cases:  

**Backward Compatibility with Hive:** LakeChime provides backward compatibility with Hive by supporting partition triggers for all table types, including modern table formats, at scale.  

**Forward Compatibility with Modern Table Formats:** LakeChime offers forward compatibility with modern table formats by facilitating snapshot trigger semantics for all table types, including the Hive table format, at scale.  

**Simpler Data Lake Migrations:** LakeChime is an essential component to unlock the migrations of data lakes from the Hive table format to the modern formats. It abstracts away the metadata implementation details, and provides a compatibility layer for the data trigger aspects through its forward and backward compatibility.  

**Benefits of Snapshot Triggers:** Snapshot triggers are a step up in UX compared to traditional partition triggers because they enable both low-latency computation and the ability to catch up on late data arrivals.  

**Incremental Compute:** LakeChime unlocks incremental compute at scale when the underlying table format supports incremental scans, bridging the gap between batch and stream processing, and paving the path to smarter and more efficient compute workflows.  

**Ease of Integration:** LakeChime is easily integrated with data producers, consumers, and data scheduling systems (e.g., [Airflow](https://airflow.apache.org/) or [dbt](https://www.getdbt.com/product/what-is-dbt?hsa_ver=3&hsa_ad=671180188489&hsa_acc=8253637521&hsa_grp=150039236302&utm_source=google&utm_source=adwords&utm_campaign=q2-2024_us-nonbrand-database-design_co&utm_campaign=us-nonbrand-database-design_co&utm_medium=paid-search&utm_medium=ppc&hsa_src=g&gad_source=1&hsa_mt=p&gclid=Cj0KCQiA2eKtBhDcARIsAEGTG43BgIefsTc0sTOvNKm1PQkkwMZMh7bAIKvYEGz_avXh5SZ_-EnCCsgaAhmXEALw_wcB&utm_content=_kw-datatool-ph___&hsa_tgt=kwd-799109805016&hsa_cam=20033614019&hsa_net=adwords&utm_term=all_na_us&utm_term=data%20build%20tool&hsa_kw=data%20build%20tool)) to trigger pipelines upon the availability of data.  

In the following sections, we explore the inner workings of LakeChime, illustrating its integration with the popular scheduling platform, Airflow. We'll also offer a comprehensive demonstration of the user experience, showcasing how LakeChime, Airflow, and Iceberg collectively facilitate incremental computing on Iceberg tables.

## Data Change Event: The Foundation of LakeChime's Data Trigger System

At the core of LakeChime's data trigger system lies the Data Change Event, or DCE. DCEs capture the concept of data changes within a data table. DCEs are registered by data producers upon updates. Data consumers, often orchestrated through frameworks like dbt or Airflow, consume these events to propagate changes downstream, which, in turn, emit new DCEs. Notably, data producers encompass a variety of platforms, including data ingestion platforms, compute engines, or table catalogs.  

Let's dissect the key specifications of a Data Change Event:  

<table><tbody><tr><td>Field Name</td><td>Type</td><td>Description</td></tr><tr><td>event_ts</td><td>Long</td><td>The Unix epoch timestamp indicating when the DCE is registered. Automatically generated at registration time.</td></tr><tr><td>table</td><td>String</td><td>The name of the table associated with the update.</td></tr><tr><td>partition</td><td>List&lt;String&gt;</td><td><p>The value of the partition that is updated. This list structure accommodates multi-level partitioning, with each list element corresponding to one partitioning level. When the table is not partitioned, this field is NULL. JSON format is employed to capture diverse partition data types, including complex ones. Supports identity partitions in the current version.</p></td></tr><tr><td>snapshot_id</td><td>String</td><td>The snapshot ID of the update, according to the table format metadata. Can be NULL for Hive tables.</td></tr><tr><td>snapshot_ts</td><td>Long</td><td>The Unix epoch timestamp of the snapshot_id, based on the table format metadata. Can be NULL for Hive tables.</td></tr><tr><td>prev_snapshot_id</td><td>String</td><td>The previous snapshot ID of the value in “snapshot_id”.</td></tr><tr><td>table_format</td><td>Enum</td><td>The table format utilized to interpret the snapshot_id. Enum values are potential table format IDs like HIVE, ICEBERG, DELTA, etc.</td></tr><tr><td>operation_type</td><td>Enum</td><td>The type of update operation. Possible values include APPEND, DELETE, or UPDATE. The UPDATE operation signifies that no assumptions can be made about the operation type, possibly encompassing APPEND, DELETE, or both actions.</td></tr><tr><td>tags</td><td>Map&lt;String, String&gt;</td><td>A map containing key-value pairs set by the producer to annotate events with domain-specific attributes. These attributes can be interpreted by consumers to facilitate tailored processing. For example, certain events may be marked to reflect hourly or daily 99% data completeness for ingested tables.</td></tr></tbody></table>

LakeChime DCEs form the foundation for supporting both partition and snapshot data triggers, as each DCE jointly captures both partition and snapshot information. Beyond offering the benefits of each trigger type for all table types, supporting these dual aspects is crucial for compatibility between traditional and modern data lake table formats. This ensures a seamless compatibility layer and facilitates smoother data lake migrations.

## Data Trigger Semantics

In this section, we describe the semantics of both types of triggers supported by LakeChime, Partition Triggers and Snapshot Triggers. Further, we show their user-facing APIs, e.g., as part of configuration parameters of scheduling platforms such as Airflow or dbt, and how to implement those APIs using LakeChime DCEs.

### Partition Triggers

In the case of partition triggers, users initiate data flows based on the availability of specific partitions within their data tables. To configure partition triggers, users supply explicit configuration parameters to the scheduler, which govern how and when data flows should be triggered. These parameters include:

-   **String table:** The name of the table that the data flow depends on.
-   **List<String> partition:** Explicit values of partitions to monitor for data availability. These values can be expressed as functions of the current timestamp, enabling dynamic scheduling.
-   **Map<String, String> tags:** Domain-specific attributes that users can employ to further filter events by. Partition checks are successful only if the respective events tags match the provided tags.  
    

Users further define the frequency at which these conditions should be evaluated, determined by the following parameters:

-   **Long StartTimestamp:** The starting timestamp from which to initiate trigger evaluations.
-   **Int Frequency:** The frequency at which trigger evaluations are performed.
-   **Enum FrequencyUnit:** The unit of time, such as minutes, hours, or days.  
    

The trigger semantics are straightforward. For each specified timestamp (defined by StartTimestamp, Frequency, FrequencyUnit), LakeChime retrieves events corresponding to the specified criteria (table, partition, tags). DCE search range is between 0 and current timestamp. A data flow is triggered if the result of this evaluation is non-empty. When triggered, the list of DCEs returned from the API is passed to the data flow as environment variables, to be used further in the data processing logic.

### Snapshot Triggers

Snapshot triggers offer a further simplified user experience by focusing solely on conditions related to tables, without the need to specify partitions. In this mode, a data flow is triggered upon a change that occurs within any partition, with execution pacing defined according to user preferences. Users configure snapshot triggers with the following parameters:

-   **String table:** The name of the table upon which the data flow depends.
-   **Map<String, String> tags:** Domain-specific attributes used to annotate events for tailored processing.

Similar to partition triggers, users set the frequency of trigger evaluations using parameters like:

-   **Long StartTimestamp.**
-   **Int Frequency.**
-   **Enum FrequencyUnit**.  
    

The trigger semantics are as follows. For each specified timestamp (defined by StartTimestamp, Frequency, FrequencyUnit), LakeChime obtains events matching the (table, tags). The DCE search range is between the last timestamp (i.e., last timestamp when the flow successfully executed) and the current timestamp. A data flow is triggered if the result of this evaluation is non-empty for any table. When a flow is triggered, the list of DCEs returned from LakeChime is made available in the form of environment variables, facilitating data computation based on data encoded in those events.  

With both partition and snapshot triggers at their disposal, users can tailor their data processing workflows to match specific requirements and preferences, ensuring efficient and precise handling of data changes within their systems.  

Putting it all together, various data sources, e.g., Iceberg clients or Hive clients, register DCEs in LakeChime upon writing data. Scheduling platforms, e.g., Airflow or dbt, query LakeChime based on trigger conditions from user flows. Once the trigger conditions are satisfied, the user flow is executed, and the list of DCEs relevant to that execution is passed from the scheduling platform to the user flow as environment variables. Figure 1 illustrates an architecture diagram of this flow. In the next section, we showcase an end-to-end data processing UX for incremental compute using Airflow, [Apache Spark](https://spark.apache.org/), Iceberg, and LakeChime.

Figure 1: LakeChime data trigger ecosystem architecture diagram

## Case Study: Incremental Compute with LakeChime, Airflow, Iceberg, and Spark

This case study demonstrates LakeChime's API integration with Airflow, emphasizing the creation of an efficient incremental data processing experience through Iceberg tables and Spark.

### LakeChime and Airflow Integration

LakeChime is integrated with Airflow through a custom DataChangeEventOperator. The DataChangeEventOperator is designed to interact with LakeChime's APIs for handling Data Change Events (DCEs). Its primary role is to query LakeChime for any changes in data within a specified interval, making it a key element in enabling incremental processing in data workflows. The signature of the DataChangeEventOperator looks like this:

```none
DataChangeEventOperator( task_id, dataset, cluster, start_time, end_time, )
```

The parameters of the DataChangeEventOperator correspond to the LakeChime API parameters for querying data changes on a table (dataset) on a specific cluster, within a time range. Under the hood, it performs the following functionalities:

-   **Querying Data Changes:** The operator is configured to target a specific table. It sends a query to the LakeChime service, requesting information about any changes that have occurred within a specified time range. This is crucial for identifying incremental changes in data, as opposed to processing the entire dataset.  
    
-   **Time Range Management:** While the DataChangeEventOperator exposes parameters to set the time range of the data change query, the best practice is to have the time range aligned with the [DAG](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html)'s execution schedule, ensuring that the operator checks for new data changes in sync with each workflow run. This can be done by utilizing Airflow's scheduling parameters, like [data\_interval\_start](https://airflow.apache.org/docs/apache-airflow/stable/templates-ref.html#data-interval-start) and [data\_interval\_end](https://airflow.apache.org/docs/apache-airflow/stable/templates-ref.html#data-interval-end).
-   **Handling LakeChime's Response:** Once the DataChangeEventOperator sends the request, LakeChime responds with a list of DCEs. These events contain detailed information about the changes in the dataset, such as the nature of the change (e.g., data addition, deletion, update), the affected partitions, and relevant metadata like timestamps and snapshot IDs.  
    
-   **Event Parsing and Encoding:** The operator processes the received DCEs, potentially encoding them for downstream tasks within the DAG. This encoding translates the detailed information from LakeChime into a format that can be efficiently consumed by subsequent tasks, like a Spark application for data processing.  
    
-   **Integration with Other Airflow Operators:** The DataChangeEventOperator works in conjunction with other Airflow operators, such as the [ShortCircuitOperator](https://registry.astronomer.io/providers/apache-airflow/versions/latest/modules/shortcircuitoperator) for handling scenarios where no data changes are detected, and the [SparkSubmitOperator](https://airflow.apache.org/docs/apache-airflow/1.10.10/_api/airflow/contrib/operators/spark_submit_operator/index.html) for executing data processing tasks based on the identified changes.  
    

### User Workflow Overview

The DAG example below showcases using the DataChangeEventOperator as a building block for defining the DAG.  

**DAG Initialization:** 

Initially, the DAG is initialized with the basic parameters like name, scheduling interval, start time, etc.

```none
with DAG( "example_dag", schedule_interval="@hourly", start_date=pendulum.datetime(2022, 6, 1, tz="America/Los_Angeles"), user_defined_macros={"encode_events": encode_events_dict} ) as dag: # DAG configuration here
```

The rest of the DAG configuration is illustrated in the following sections.  

**Incorporating LakeChime's Incremental Triggers:**

Once the DAG is declared, the DataChangeEventOperator can be leveraged to fetch the relevant events from LakeChime.

```none
dataset_name = 'data.pageviews' fetched_events = DataChangeEventOperator( task_id="fetched_events", dataset=dataset_name, cluster=DataChangeEvent.CLUSTER1, start_time="{{ data_interval_start.int_timestamp * 1000 }}", end_time="{{ data_interval_end.int_timestamp * 1000 }}", )
```

**Handling of No-Change Scenarios:**

When no events are fetched in fetched\_events, the next steps of the DAG are short circuited so the data processing job is not submitted for execution. This avoids unnecessary execution of a Spark job when no change in upstream data has taken place.

```none
skip_downstream_when_no_events = ShortCircuitOperator( task_id="skip_downstream_when_no_events", python_callable=partial(events_exist, "fetched_events"), provide_context=True, )
```

**Processing the Data Changes:**

The next step is to submit the Spark application. fetched\_events is passed to the SparkSubmitOperator through Airflow’s xcom\_pull mechanism. A utility method encode\_events prepares the events for the Spark application processing.

```none
spark_submit_task = SparkSubmitOperator( task_id="count_rows", class_name="com.linkedin.testspark.HelloLakeChime", dependency_ivy_list=["com.linkedin.lakechime-spark:lakechime-spark-impl_2.12:0.0.+"], args=[dataset_name, '{{ encode_events(task_instance.xcom_pull(task_ids="fetched_events")) }}'], )
```

### Efficient Incremental Processing with Spark on Iceberg Tables

Building upon the LakeChime-Airflow integration, we continue our UX illustration with how this setup enables effective incremental processing in Spark, particularly with Iceberg tables. In contrast to Hive tables, where processing often involves redoing tasks (like repeatedly scanning the same set of data), the Iceberg format allows for querying the exact changes or deltas between data snapshots. This capability is pivotal for incremental computing, as it enables processing only the modified portions of the data. We describe the steps in the Spark side of the workflow below.  

**Identifying the Snapshot Range:**

The Spark job begins by obtaining the snapshot range for the incremental scan from the list of DCEs passed from Airflow. This is done by chaining snapshot IDs of the records (DCEs) using the prev\_snapshot\_id field. Once a full chain is constructed, say starting at firstDce and ending at lastDce, the job identifies the range of snapshots to process, basically, firstDce.getPrevSnapshotId, and lastDce.getSnapshotId.  

**Delta Processing:**

With the snapshot range determined, the Spark job processes the delta of the data. It reads the data between the previous and current snapshot IDs, focusing only on the changes.

```none
val df = spark.read .format("catalog-name") // Optional, depending on the format .option("start-snapshot-id", firstDce.getPrevSnapshotId) .option("end-snapshot-id", lastDce.getSnapshotId) .load(firstDce.getTableName) // Further processing logic goes here
```

By processing only the deltas between snapshots, the Spark job significantly reduces the amount of data to be processed, leading to faster and more efficient computations. Further, the approach is more scalable compared to the non-incremental approach, as it is capable of handling large datasets and frequent updates more efficiently.  

**Conclusion:**

This Spark-based incremental processing, when combined with LakeChime's data management and Airflow's orchestration, exemplifies a sophisticated data processing paradigm. It allows users to efficiently process only the necessary data changes, reducing resource usage and improving processing times. This integration of technologies showcases the potential for advanced data processing strategies in modern data ecosystems.

## Next Steps  

As we continue to develop and refine LakeChime, we are excited to share the upcoming milestones in its journey.  

Our next move is to integrate LakeChime with dbt and [Coral](https://github.com/linkedin/coral). This integration is focused on automating the maintenance of incremental views, a key aspect of efficient data processing. While LakeChime provides data application developers the opportunity to build incremental pipelines, developers are still required to implement the incremental processing logic, which is usually hard to formulate. With the dbt integration through Coral, developers can be freed from having to devise the incremental logic, and rather express their logic in batch semantics. Behind the scenes, Coral transforms the logic and executes it incrementally on the target execution engine. A demonstration of this end-to-end integration can be found in the [Coral-dbt code module](https://github.com/linkedin/coral/tree/master/coral-dbt), and its respective slide deck. Stay tuned for more updates as we embark on these exciting new phases in LakeChime's journey.

## Acknowledgements

LakeChime is a product of collaboration of multiple teams at LinkedIn. LakeChime could not reach its current form without the contributions of [Janki Akhani](https://www.linkedin.com/in/jakhani/), [Trevor Devore](https://www.linkedin.com/in/tdevore7/), [Zihan Li](https://www.linkedin.com/in/zihan-li-0a8a15149/), [Jack Moseley](https://www.linkedin.com/in/jack-moseley-2b8916121/), [Ratandeep Ratti](https://www.linkedin.com/in/rdsr13/), [Prasad Karkera](https://www.linkedin.com/in/prasadkarkera/), [Lei Sun](https://www.linkedin.com/in/lei-s-a93138a0/), [Sushant Raikar](https://www.linkedin.com/in/sushantraikar/), [Rohit Kumar](https://www.linkedin.com/in/rkumar2506/), [Abhishek Nath](https://www.linkedin.com/in/abhishek-nath-8b691213/), [Sumedh Sakdeo](https://www.linkedin.com/in/sumedhsakdeo/), [Tanzir Musabbir](https://www.linkedin.com/in/tanzir/), [Manisha Kamal](https://www.linkedin.com/in/manishakamal/), [Swathi Koundinya](https://www.linkedin.com/in/swathikoundinya/), [Abhishek Tiwari](https://www.linkedin.com/in/findabti/), and [Kamal Duggireddy](https://www.linkedin.com/in/kamaldr/) and the support of [Sumitha Poornachandran](https://www.linkedin.com/in/sumitha-poornachandran-097a6019/) (Alumni), [Renu Tewari](https://www.linkedin.com/in/renutewari/) (Alumni), and [Kartik Paramasivam](https://www.linkedin.com/in/kartik-paramasivam-b71b0711/).

Topics: [Data](https://www.linkedin.com/blog/engineering/data) [Data Management](https://www.linkedin.com/blog/engineering/data-management)

Related articles