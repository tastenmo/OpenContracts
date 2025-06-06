# Generated by Django 4.2.20 on 2025-03-12 04:29

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import pgvector.django.vector


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("documents", "0014_document_description_embedding"),
        ("annotations", "0029_alter_note_creator"),
    ]

    operations = [
        migrations.CreateModel(
            name="Embedding",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("backend_lock", models.BooleanField(default=False)),
                ("is_public", models.BooleanField(default=False)),
                (
                    "embedder_path",
                    models.CharField(
                        blank=True,
                        help_text="Identifier for the embedding model or pipeline used (e.g. 'openai/text-embedding-ada-002').",
                        max_length=256,
                        null=True,
                    ),
                ),
                (
                    "vector_384",
                    pgvector.django.vector.VectorField(
                        blank=True, dimensions=384, null=True
                    ),
                ),
                (
                    "vector_768",
                    pgvector.django.vector.VectorField(
                        blank=True, dimensions=768, null=True
                    ),
                ),
                (
                    "vector_1536",
                    pgvector.django.vector.VectorField(
                        blank=True, dimensions=1536, null=True
                    ),
                ),
                (
                    "vector_3072",
                    pgvector.django.vector.VectorField(
                        blank=True, dimensions=3072, null=True
                    ),
                ),
                (
                    "created",
                    models.DateTimeField(blank=True, default=django.utils.timezone.now),
                ),
                (
                    "modified",
                    models.DateTimeField(blank=True, default=django.utils.timezone.now),
                ),
                (
                    "annotation",
                    models.ForeignKey(
                        blank=True,
                        help_text="References the Annotation that this embedding belongs to (if any).",
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="embedding_set",
                        to="annotations.annotation",
                    ),
                ),
                (
                    "creator",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "document",
                    models.ForeignKey(
                        blank=True,
                        help_text="References the Document that this embedding belongs to (if any).",
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="embedding_set",
                        to="documents.document",
                    ),
                ),
                (
                    "note",
                    models.ForeignKey(
                        blank=True,
                        help_text="References the Note that this embedding belongs to (if any).",
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="embedding_set",
                        to="annotations.note",
                    ),
                ),
                (
                    "user_lock",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="locked_%(class)s_objects",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "Embedding",
                "verbose_name_plural": "Embeddings",
            },
        ),
        migrations.AddField(
            model_name="annotation",
            name="embeddings",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="annotations",
                to="annotations.embedding",
            ),
        ),
        migrations.AddField(
            model_name="note",
            name="embeddings",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="notes",
                to="annotations.embedding",
            ),
        ),
        migrations.AddIndex(
            model_name="embedding",
            index=models.Index(
                fields=["embedder_path"], name="annotations_embedde_200067_idx"
            ),
        ),
        migrations.AddIndex(
            model_name="embedding",
            index=models.Index(
                fields=["created"], name="annotations_created_943733_idx"
            ),
        ),
        migrations.AddIndex(
            model_name="embedding",
            index=models.Index(
                fields=["modified"], name="annotations_modifie_037252_idx"
            ),
        ),
    ]
